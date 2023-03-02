from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.configuration_utils import register_to_config

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import StrictInt, StrictFloat, StrictBool, StrictStr

import torch
import torch.utils.checkpoint
import torch.nn.functional as F

from ldm.modules.diffusionmodules.positional_encoding import SinusoidalPositionalEmbedding


class UNet2DConditionPatchModel(UNet2DConditionModel):
    @register_to_config
    def __init__(
        self,

        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        time_embedding_type: str = "positional",  # fourier, positional
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,

        padding_idx: StrictInt = 0,
        init_size: StrictInt = 128,
        div_half_dim: StrictBool = False,
        center_shift: StrictInt = 64,
        interpolation_mode: StrictStr = "bilinear",
    ):
        super().__init__(sample_size=sample_size,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         center_input_sample=center_input_sample,
                         flip_sin_to_cos=flip_sin_to_cos,
                         freq_shift=freq_shift,
                         down_block_types=down_block_types,
                         mid_block_type=mid_block_type,
                         up_block_types=up_block_types,
                         only_cross_attention=only_cross_attention,
                         block_out_channels=block_out_channels,
                         layers_per_block=layers_per_block,
                         downsample_padding=downsample_padding,
                         mid_block_scale_factor=mid_block_scale_factor,
                         act_fn=act_fn,
                         norm_num_groups=norm_num_groups,
                         norm_eps=norm_eps,
                         cross_attention_dim=cross_attention_dim,
                         attention_head_dim=attention_head_dim,
                         dual_cross_attention=dual_cross_attention,
                         use_linear_projection=use_linear_projection,
                         class_embed_type=class_embed_type,
                         num_class_embeds=num_class_embeds,
                         upcast_attention=upcast_attention,
                         resnet_time_scale_shift=resnet_time_scale_shift,
                         time_embedding_type=time_embedding_type,  # fourier, positional
                         timestep_post_act=timestep_post_act,
                         time_cond_proj_dim=time_cond_proj_dim,
                         conv_in_kernel=conv_in_kernel,
                         conv_out_kernel=conv_out_kernel,
                         projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
                         )
        assert block_out_channels[0] % 2 == 0
        self.head_position_encode = SinusoidalPositionalEmbedding(embedding_dim=block_out_channels[0]//2,
                                                                  padding_idx=padding_idx,
                                                                  init_size=init_size,
                                                                  div_half_dim=div_half_dim,
                                                                  center_shift=center_shift)
        self.init_size = init_size
        self.interpolation_mode = interpolation_mode

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        crop_boxes: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.

        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        head_grid = self.head_position_encode(torch.ones([sample.shape[0], sample.shape[1], self.init_size, self.init_size], dtype=self.dtype,
                                                         device=sample.device))

        if crop_boxes is not None:

            head_grid = torch.cat([F.interpolate(hg.unsqueeze(0)[:, :, box[0]: box[2], box[1]: box[3]],
                                                 (sample.shape[2], sample.shape[3]), mode='bilinear', align_corners=True)
                                   for hg, box in
                                   zip(head_grid, crop_boxes)], dim=0)
        else:
            head_grid = F.interpolate(
                head_grid, (sample.shape[2], sample.shape[3]), mode='bilinear', align_corners=True)

        sample += head_grid

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(
                upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


if __name__ == "__main__":
    unet = UNet2DConditionPatchModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", revision=None, low_cpu_mem_usage=False, device_map=None
    )