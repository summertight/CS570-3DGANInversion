# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from training.vggloss import VGGLoss
from training.idloss import IDLoss

from training.DECA.decalib.deca import DECA
from training.DECA.decalib.utils.tensor_cropper import Cropper
from training.DECA.decalib.datasets.detectors import batch_FAN
from training.DECA.decalib.utils.config import cfg as deca_cfg

from training.mobile_face_net import load_face_landmark_detector

def denorm(x):
    return (x / 2 + 0.5).clamp(0, 1)

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased'):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
class StyleGAN2AELoss(Loss):
    def __init__(self, device, rank, G, D, E, augment_pipe=None, blur_init_sigma=0, blur_fade_kimg=0, neural_rendering_resolution_initial=64, \
    neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, filter_mode='antialiased', n_random_labels=None,\
     loss_selection = None,  invert_map=False, resolution_encode = 512,\
     r1_gamma_fade_kimg=0,pl_decay=0.01, r1_gamma=10,style_mixing_prob=0,pl_weight=0, pl_batch_shrink=2,pl_no_weight_grad=False,r1_gamma_init=0,\
     gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, \
     ):

     
        super().__init__()
        self.device             = device
        self.rank               = rank
        self.G                  = G
        self.D                  = D
        self.E                  = E
        self.augment_pipe       = augment_pipe
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.pl_mean            = torch.zeros([], device=device)
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.filter_mode        = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
   
      
        self.r1_gamma           = r1_gamma

        self.loss_selection = loss_selection
        


        #XXX

        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad

        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.blur_raw_target = True
        #XXX

        self.invert_map = invert_map
        if 'vgg' in loss_selection:
            self.criterion_VGG = VGGLoss().to(rank)
        # TODO: Commit ID Loss
        if 'id' in loss_selection:
            self.criterion_ID  = IDLoss().to(rank)
        if 'deca' in loss_selection:
            self.criterion_DECA = DECA(config=deca_cfg, device=rank)
            self.landmark_detector = load_face_landmark_detector()
            self.landmark_detector = self.landmark_detector.to(rank)
            self.landmark_detector.eval()
        #breakpoint()
        self.resolution_encode = resolution_encode
    
    def run_EG(self, img_src, c, neural_rendering_resolution, h_start=None):
        #breakpoint()

        #img_src_resized = filtered_resizing(img_src, size=self.resolution_encode, f=self.resample_filter, filter_mode=self.filter_mode)
        if self.invert_map:

            
            ws, maps = self.E(img_src)

            #breakpoint()
            
            gen_output = self.G.synthesis(ws, maps, c, neural_rendering_resolution=neural_rendering_resolution)
        else:
            
            ws = self.E(img_src)
            gen_output = self.G.synthesis(ws, None, c, neural_rendering_resolution=neural_rendering_resolution)
            if False:
                c_roll = torch.roll(c.clone(), 1, 0)
                gen_output_roll = self.G.synthesis(ws, None, c_roll, neural_rendering_resolution=neural_rendering_resolution)
                return gen_output, gen_output_roll
        return gen_output


    def run_D(self, img, c, blur_sigma=0):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                               torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                               dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c)
        return logits



    def accumulate_gradients(self, phase, img_src, c, gain, cur_nimg ):
        assert phase in ['Gmain', 'Dmain']
        
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        img_src_raw = filtered_resizing(img_src, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
        #img_drv_raw = filtered_resizing(img_drv, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
        img_src_resized = filtered_resizing(img_src, size=self.resolution_encode, f=self.resample_filter, filter_mode=self.filter_mode)
        real_img = {'image_src': img_src, 'image_src_raw': img_src_raw, 'image_src_resized': img_src_resized}
        
        # Gmain: L1 Loss and VGG Loss. / Maximize logits for generated images.
        if phase in ['Gmain']:
            with torch.autograd.profiler.record_function('Gmain_forward'):

                gen_img = self.run_EG(real_img['image_src_resized'], c, neural_rendering_resolution=neural_rendering_resolution)
                #gen_img = self.run_EG(real_img['image_src'], c, neural_rendering_resolution=neural_rendering_resolution)

                # L1 Loss
                loss_Gmain = 0.0
                
                if 'l1' in self.loss_selection:
          
                    #import pdb;pdb.set_trace()
                    loss_l2 = torch.nn.functional.l1_loss(gen_img['image'], real_img['image_src'])
                    loss_l2_raw = torch.nn.functional.l1_loss(gen_img['image_raw'], real_img['image_src_raw'])

                    loss_Gmain += self.loss_selection['l1']*(loss_l2_raw + loss_l2)

                    training_stats.report('Loss/loss_L1', loss_l2)
                    training_stats.report('Loss/loss_L1_raw', loss_l2_raw)
                
                # Perceptual Loss
                if 'vgg' in self.loss_selection:
                
                    
                    loss_vgg = self.criterion_VGG(gen_img['image'], real_img['image_src']).mean()
                    
                    loss_Gmain += self.loss_selection['vgg'] * loss_vgg

                    training_stats.report('Loss/loss_VGG', loss_vgg)
                    
                
                # TODO: Commit ID Loss
                if 'id' in self.loss_selection:
                    
                    loss_id = self.criterion_ID(gen_img['image'], real_img['image_src']).mean()
                    loss_Gmain += self.loss_selection['id'] * loss_id
                    training_stats.report('Loss/loss_ID', loss_id)
                
                if 'deca' in self.loss_selection:
                    
                    ldmks_gen = self.FAN.run((gen_img['image']+1)*127.5)#XXX (0,255)
                    #import pdb;pdb.set_trace()

                    image_gen_cropped = self.landmark_detector.align_face(
                        inputs=denorm(gen_img['image']), scale=1.25, inverse=False, target_size=224)
                    
                    #image_cropped_gen, _ = self.cropper.crop(image = gen_img['image'],points = torch.tensor(ldmks_gen))
                
                    codedict_gen = self.criterion_DECA.encode(image_gen_cropped)#XXX (0,1)

                    image_src_cropped = self.landmark_detector.align_face(
                        inputs=denorm(real_img['image_src']), scale=1.25, inverse=False, target_size=224)
                    codedict_src = self.criterion_DECA.encode(image_src_cropped)
                    loss_3dmm_shape = torch.nn.functional.mse_loss(codedict_gen['shape'], codedict_src['shape'])
                    loss_3dmm_exp = torch.nn.functional.mse_loss(codedict_gen['exp'], codedict_src['exp'])
                    loss_Gmain += (loss_3dmm_shape + loss_3dmm_exp)
                    training_stats.report('Loss/LIA3D/loss_3dmm_shape', loss_3dmm_shape)
                    training_stats.report('Loss/LIA3D/loss_3dmm_exp', loss_3dmm_exp)

                    
                if 'gan' in self.loss_selection:
                
                        
                    gen_logits = self.run_D(gen_img, c, blur_sigma=blur_sigma)
                    loss_gan = torch.nn.functional.softplus(-gen_logits)
                    loss_Gmain += loss_gan.mean()
                    training_stats.report('Loss/LIA3D/loss_ganG', loss_gan.mean())
     
                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()


        # Dmain: Minimize logits for generated images.
        if 'gan' in self.loss_selection:

            loss_Dgen = 0
            if phase in ['Dmain']:
                with torch.autograd.profiler.record_function('Dgen_forward'):
                    if self.encoder_type=='self_cross':
                        gen_img = self.run_G_double(real_img['image_src'], real_img['image_drv'], c, neural_rendering_resolution=neural_rendering_resolution)
                        gen_logits = self.run_D_double(gen_img, c, blur_sigma=blur_sigma)

                        training_stats.report('Loss/scores/fake_self', gen_logits[0])
                        training_stats.report('Loss/signs/fake_self', gen_logits[0].sign())
                        training_stats.report('Loss/scores/fake_cross', gen_logits[1])
                        training_stats.report('Loss/signs/fake_cross', gen_logits[1].sign())
                        loss_Dgen_self = torch.nn.functional.softplus(gen_logits[0])
                        loss_Dgen_cross = torch.nn.functional.softplus(gen_logits[1])
                        loss_Dgen = loss_Dgen_self+loss_Dgen_cross

                    else:
                            
                        gen_img = self.run_G(real_img['image_src'], real_img['image_drv'], c, neural_rendering_resolution=neural_rendering_resolution)
                        gen_logits = self.run_D(gen_img, c, blur_sigma=blur_sigma)

                        training_stats.report('Loss/scores/fake', gen_logits)
                        training_stats.report('Loss/signs/fake', gen_logits.sign())
                        
                        loss_Dgen = torch.nn.functional.softplus(gen_logits)
                
                with torch.autograd.profiler.record_function('Dgen_backward'):
                    
                    loss_Dgen.mean().mul(gain).backward()
            # Dmain: Maximize logits for real images.
            if phase in ['Dmain']:
                with torch.autograd.profiler.record_function('Dreal_forward'):
                    real_img_tmp_image_self = real_img['image_drv'].detach().requires_grad_(False)
                    real_img_tmp_image_raw_self = real_img['image_drv_raw'].detach().requires_grad_(False)
                    real_img_tmp_image_cross = real_img['image_drv'].detach().requires_grad_(False)
                    real_img_tmp_image_raw_cross = real_img['image_drv_raw'].detach().requires_grad_(False)
                    real_img_tmp = {'image': real_img_tmp_image_self, 'image_raw': real_img_tmp_image_raw_self, 'image_cross':real_img_tmp_image_cross,'image_raw_cross':real_img_tmp_image_raw_cross}

                    real_logits = self.run_D_double(real_img_tmp, c, blur_sigma=blur_sigma)
                    
                    loss_Dreal = 0
                    #loss_Dreal = torch.nn.functional.softplus(-real_logits)

                    loss_Dreal_self = torch.nn.functional.softplus(-real_logits[0])
                    loss_Dreal_cross = torch.nn.functional.softplus(-real_logits[1])
                    loss_Dreal = loss_Dreal_self+loss_Dreal_cross
                    
                    training_stats.report('Loss/scores/real_self', real_logits[0])
                    training_stats.report('Loss/signs/real_self', real_logits[0].sign())
                    training_stats.report('Loss/scores/real_fake', real_logits[1])
                    training_stats.report('Loss/signs/real_fake', real_logits[1].sign())
                 
                    if self.encoder_type =='self_cross':
                        training_stats.report('Loss/LIA3D/loss_ganD', loss_Dgen_self.mean() +loss_Dgen_cross.mean() + loss_Dreal.mean())

                    else:
                        training_stats.report('Loss/LIA3D/loss_ganD', loss_Dgen + loss_Dreal)

                with torch.autograd.profiler.record_function('Dreal_backward'):
                    (loss_Dreal).mean().mul(gain).backward()
        
        return real_img, gen_img