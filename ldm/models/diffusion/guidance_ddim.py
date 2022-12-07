"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


class DDIMSingleSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        def to_torch(x): return x.clone().detach().to(
            torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(
            self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(
            np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(
            np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):
        timesteps = np.arange(
            self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full(
                (x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec


class DDIMMultiSampler(DDIMSingleSampler):
    def __init__(self, model, guide_model_list, schedule="linear", **kwargs):
        super().__init__(model=model, schedule=schedule, **kwargs)
        self.guide_model_list = guide_model_list
        self.num_guidance = len(guide_model_list)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               conditioning_single_list=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               unconditional_conditioning_single_list=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        if conditioning_single_list is not None:
            assert isinstance(conditioning_single_list, list)
            assert len(conditioning_single_list) == self.num_guidance
            for conditioning_single in conditioning_single_list:
                if isinstance(conditioning_single, dict):
                    cbs = conditioning_single[list(
                        conditioning_single.keys())[0]].shape[0]
                    if cbs != batch_size:
                        print(
                            f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
                else:
                    if conditioning_single.shape[0] != batch_size:
                        print(
                            f"Warning: Got {conditioning_single.shape[0]} conditionings but batch-size is {batch_size}")
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, conditioning_single_list, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    unconditional_conditioning_single_list=unconditional_conditioning_single_list,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, cond_sin_list, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      unconditional_conditioning_single_list=None, crop_boxes=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(
                timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(
            range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                print(f"Masking with {mask.shape}")
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
            range_t_max_list = [model.extra_config['range_t_max']
                                for model in self.guide_model_list]
            range_t_min_list = [model.extra_config['range_t_min']
                                for model in self.guide_model_list]

            img_sin_list = [img] * self.num_guidance
            outs = self.p_sample_ddim(img, img_sin_list, cond, cond_sin_list, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      unconditional_conditioning_single_list=unconditional_conditioning_single_list,
                                      crop_boxes=crop_boxes,
                                      single_guidance_list=[(range_t_max >= step >= range_t_min) for range_t_max, range_t_min in zip(range_t_max_list, range_t_min_list)],)
            img, pred_x0, img_sin_list, pred_x0_sin_list = outs

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, x_sin_list, c, c_sin_list, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, crop_boxes=None,
                      single_guidance_list=None, unconditional_conditioning_single_list=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, crop_boxes=crop_boxes)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if crop_boxes is not None:
                crop_boxes = torch.cat([crop_boxes] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(
                x_in, t_in, c_in, crop_boxes=crop_boxes).chunk(2)

            guidance = torch.zeros_like(e_t)
            beta = 0
            e_t_sin_list = list()

            for x_sin, c_sin, unconditional_conditioning_single, model_sin, single_guidance in zip(x_sin_list, c_sin_list, unconditional_conditioning_single_list, self.guide_model_list, single_guidance_list):

                x_in_sin = torch.cat([x_sin] * 2)
                c_in_sin = torch.cat(
                    [unconditional_conditioning_single, c_sin])
                e_t_uncond_sin, e_t_sin = model_sin.apply_model(
                    x_in_sin, t_in, c_in_sin, crop_boxes=crop_boxes).chunk(2)
                beta_sin = model_sin.extra_config['beta']
                if single_guidance:
                    beta += beta_sin
                    guidance += beta_sin * e_t_sin

                e_t_sin_list.append(
                    e_t_uncond_sin + unconditional_guidance_scale * (e_t_sin - e_t_uncond_sin))
            # assert 0 <= beta <= 1

            e_t = e_t_uncond + unconditional_guidance_scale * \
                (e_t * (1. - beta) + guidance - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs)
            e_t_sin_list = [score_corrector.modify_score(model_sin, e_t_sin, x_sin, t, c_sin, **corrector_kwargs)
                            for model_sin, e_t_sin, x_sin, c_sin in zip(self.guide_model_list, e_t_sin_list, x_sin_list, c_sin_list)]

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        pred_x0_sin_list = [(x_sin - sqrt_one_minus_at * e_t_sin) / a_t.sqrt()
                            for x_sin, e_t_sin in zip(x_sin_list, e_t_sin_list)]

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            pred_x0_sin_list = [model_sin.first_stage_model.quantize(pred_x0_sin)[
                0] for model_sin, pred_x0_sin in zip(self.guide_model_list, pred_x0_sin_list)]
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        dir_xt_sin_list = [(1. - a_prev - sigma_t**2).sqrt()
                           * e_t_sin for e_t_sin in e_t_sin_list]

        noise = sigma_t * \
            noise_like(x.shape, device, repeat_noise) * temperature
        noise_sin_list = [
            sigma_t * noise_like(x_sin.shape, device, repeat_noise) * temperature for x_sin in x_sin_list]
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            noise_sin_list = [torch.nn.functional.dropout(
                noise_sin, p=noise_dropout) for noise_sin in noise_sin_list]
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev_sin_list = [a_prev.sqrt() * pred_x0_sin + dir_xt_sin + noise_sin for pred_x0_sin,
                           dir_xt_sin, noise_sin in zip(pred_x0_sin_list, dir_xt_sin_list, noise_sin_list)]

        return x_prev, pred_x0, x_prev_sin_list, pred_x0_sin_list


class DDIMSinSampler(DDIMSingleSampler):
    def __init__(self, model, model_sin, schedule="linear", **kwargs):
        super().__init__(model=model, schedule=schedule, **kwargs)
        self.model_sin = model_sin

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               conditioning_single=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               unconditional_conditioning_single=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")
        if conditioning_single is not None:
            if isinstance(conditioning_single, dict):
                cbs = conditioning_single[list(
                    conditioning_single.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(
                        f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning_single.shape[0] != batch_size:
                    print(
                        f"Warning: Got {conditioning_single.shape[0]} conditionings but batch-size is {batch_size}")
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, conditioning_single, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    unconditional_conditioning_single=unconditional_conditioning_single,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, cond_sin, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      unconditional_conditioning_single=None, crop_boxes=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
            img_sin = torch.randn(shape, device=device)
        else:
            img = x_T
            img_sin = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(
                timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [
            img], 'x_inter_sin': [img_sin], 'pred_x0_sin': [img_sin]}
        time_range = reversed(
            range(0, timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                print(f"Masking with {mask.shape}")
                assert x0 is not None
                # TODO: deterministic forward pass?
                img_orig = self.model.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img
            range_t_max = self.model.extra_config['range_t_max']
            range_t_min = self.model.extra_config['range_t_min']

            img_sin = img
            outs = self.p_sample_ddim(img, img_sin, cond, cond_sin, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      unconditional_conditioning_single=unconditional_conditioning_single,
                                      crop_boxes=crop_boxes,
                                      single_guidance=(range_t_max >= step >= range_t_min))
            img, pred_x0, img_sin, pred_x0_sin = outs

            if callback:
                callback(i)
            if img_callback:
                img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)
                intermediates['x_inter_sin'].append(img_sin)
                intermediates['pred_x0_sin'].append(pred_x0_sin)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, x_sin, c, c_sin, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, crop_boxes=None,
                      single_guidance=False, unconditional_conditioning_single=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, crop_boxes=crop_boxes)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if crop_boxes is not None:
                crop_boxes = torch.cat([crop_boxes] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(
                x_in, t_in, c_in, crop_boxes=crop_boxes).chunk(2)

            x_in_sin = torch.cat([x_sin] * 2)
            c_in_sin = torch.cat([unconditional_conditioning_single, c_sin])
            e_t_uncond_sin, e_t_sin = self.model_sin.apply_model(
                x_in_sin, t_in, c_in_sin, crop_boxes=crop_boxes).chunk(2)

            beta1 = self.model.extra_config['cond_beta']
            beta2 = self.model.extra_config['cond_beta_sin']

            # e_t_sin = e_t_uncond_sin + unconditional_guidance_scale * (e_t_sin - e_t_uncond_sin)
            if single_guidance:
                e_t = e_t_uncond + unconditional_guidance_scale * \
                    (e_t * beta1 + e_t_sin * beta2 - e_t_uncond)
            else:
                e_t = e_t_uncond + unconditional_guidance_scale * \
                    (e_t - e_t_uncond)
            e_t_sin = e_t_uncond_sin + unconditional_guidance_scale * \
                (e_t_sin - e_t_uncond_sin)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(
                self.model, e_t, x, t, c, **corrector_kwargs)
            e_t_sin = score_corrector.modify_score(
                self.model_sin, e_t_sin, x_sin, t, c_sin, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        pred_x0_sin = (x_sin - sqrt_one_minus_at * e_t_sin) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            pred_x0_sin, _, * \
                _ = self.model_sin.first_stage_model.quantize(pred_x0_sin)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        dir_xt_sin = (1. - a_prev - sigma_t**2).sqrt() * e_t_sin
        noise = sigma_t * \
            noise_like(x.shape, device, repeat_noise) * temperature
        noise_sin = sigma_t * \
            noise_like(x_sin.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            noise_sin = torch.nn.functional.dropout(noise_sin, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev_sin = a_prev.sqrt() * pred_x0_sin + dir_xt_sin + noise_sin
        return x_prev, pred_x0, x_prev_sin, pred_x0_sin
