from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import torch
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from transformers import AutoTokenizer, PretrainedConfig
from typing import Any, Callable, Dict, List, Optional, Union
import importlib
import os 
import diffusers
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
import argparse
from accelerate.utils import ProjectConfiguration, set_seed



def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
    
    
class StableDiffusionGuidancePipeline(StableDiffusionPipeline):
    text_encoder_orig = None
    unet_orig = None
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker,
    ):
        super().__init__(vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            requires_safety_checker,)
        self.config['unet'] = (unet.__module__, unet.config['_class_name'])
        
        
    def add_pretrained_model(self, text_encoder, unet):
        self.text_encoder_orig = text_encoder.to(self._execution_device)
        self.unet_orig = unet.to(self._execution_device)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        return_cached_folder = kwargs.pop("return_cached_folder", False)

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        cached_folder = pretrained_model_name_or_path

        config_dict = cls.load_config(cached_folder)
        
        if config_dict['unet'][0] is None:
            config_dict['unet'][0] = 'diffusers_models'

        # 2. Load the pipeline class
        pipeline_class = cls

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs
        init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if isinstance(value, bool):
                return False
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # import it here to avoid circular import
        from diffusers import pipelines

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            # 3.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            if class_name.startswith("Flax"):
                class_name = class_name[4:]

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                # pass

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = getattr(pipeline_module, class_name)
            else:
                # else we just import it from the library.
                # NOTE: here I reuse library_name as the module name
                library = importlib.import_module(library_name)
                class_obj = getattr(library, class_name)
            
            if loaded_sub_model is None:
                load_method_name = 'from_pretrained'

                load_method = getattr(class_obj, load_method_name)
                loading_kwargs = {}

                if issubclass(class_obj, torch.nn.Module):
                    loading_kwargs["torch_dtype"] = torch_dtype
                # if issubclass(class_obj, diffusers.OnnxRuntimeModel):
                #     loading_kwargs["provider"] = provider
                #     loading_kwargs["sess_options"] = sess_options

                is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)

                # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
                # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
                # This makes sure that the weights won't be initialized which significantly speeds up loading.
                if is_diffusers_model:
                    loading_kwargs["device_map"] = device_map
                    loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

                # check if the module is in a subdirectory
                if os.path.isdir(os.path.join(cached_folder, name)):
                    loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
                else:
                    # else load from the root directory
                    loaded_sub_model = load_method(cached_folder, **loading_kwargs)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        init_kwargs['requires_safety_checker'] = False
        # 4. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 5. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        if return_cached_folder:
            return model, cached_folder
        return model
        
    def _encode_prompt_orig(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )

            if hasattr(self.text_encoder_orig.config, "use_attention_mask") and self.text_encoder_orig.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder_orig(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_orig.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder_orig.config, "use_attention_mask") and self.text_encoder_orig.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder_orig(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_orig.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_edit: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        model_based_guidance_scale: float = 0.0,
        K_min: int = 400,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )
        self.check_inputs(
            prompt_edit, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            assert isinstance(prompt_edit, str)
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            assert isinstance(prompt_edit, list)
            assert len(prompt_edit) == len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        prompt_embdes_edit = self._encode_prompt_orig(
            prompt_edit,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet_orig(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embdes_edit,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                if t > K_min and model_based_guidance_scale > 0.0:
                     
                    noise_pred_guidance = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    if do_classifier_free_guidance:
                        noise_pred_guidance_uncond, noise_pred_guidance_text = noise_pred_guidance.chunk(2)
                        noise_pred_text = noise_pred_text * (1 - model_based_guidance_scale) + noise_pred_guidance_text * model_based_guidance_scale
                    else:
                        noise_pred = noise_pred * (1 - model_based_guidance_scale) + noise_pred_guidance * model_based_guidance_scale
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)
            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
    
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        required=True,
        help='Text prompt for the fine-tuned model.'
    )
    parser.add_argument(
        '--editing_prompt',
        type=str,
        default=None,
        required=True,
        help='Text prompt for the pre-trained model.'
    )
    parser.add_argument("--seed", type=int, default=412441,
                        help="A seed for reproducible sampling.")
    parser.add_argument('--num_images_per_prompt', type=int, default=2, help='Batch size.')
    parser.add_argument('--num_iterations', type=int, default=1,)
    parser.add_argument('--model_based_guidance_scale', type=float, default=0.3, help='Scale of model-based guidance.')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Scale of classifier-free guidance.')
    parser.add_argument('--K', default=400, type=int, help='step to stop guidance')
    parser.add_argument('--ddim_steps', default=100, type=int, help='Number of ddim steps')
    parser.add_argument('--height', default=512, type=int, help='Height of the image')
    parser.add_argument('--width', default=512, type=int, help='Width of the image')
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

    
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=None
        )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=None
    )
    
    
    model_id = args.pretrained_model_name_or_path
    pipe = StableDiffusionGuidancePipeline.from_pretrained(model_id, torch_dtype=torch.float).to("cuda")
    pipe.add_pretrained_model(text_encoder=text_encoder, unet=unet)

    prompt = args.prompt
    prompt_edit = args.editing_prompt
    file_name = (prompt + '[SEP]' + prompt_edit).replace(' ', '_')
    for i in range(args.num_iterations):
        images = pipe(prompt=prompt, prompt_edit=prompt_edit,
                    model_based_guidance_scale=args.model_based_guidance_scale, 
                    K_min=args.K, num_inference_steps=args.ddim_steps, 
                    guidance_scale=args.guidance_scale, 
                    num_images_per_prompt=args.num_images_per_prompt, height=args.height, width=args.width).images

        for j, image in enumerate(images):

            image.save("{}/{}_{}.png".format(args.pretrained_model_name_or_path, file_name, i * args.num_images_per_prompt + j))