# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from camera_utils import LookAtPoseSampler
from training.crosssection_utils import sample_cross_section
from training.mapper import ExpMappingNet 


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return

    
def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output
#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = list(label_groups.keys())
        rnd.shuffle(label_order)
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    E_kwargs                = {},
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    E_opt_kwargs            = {},
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    mode                    = 'AE',
    pretrain = None,
    loss_selection = None,
    invert_map = False,
    resolution_encode = False,
    start_from_avg =False,

):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.register_buffer('dataset_label_std', torch.tensor(training_set.get_label_std()).to(device))
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    if start_from_avg:
        G.w_avg = G.backbone.mapping.w_avg.unsqueeze(0).unsqueeze(1).repeat(1,14,1).detach().cpu()
    if mode=="MFIM" or mode == 'AE' or mode == 'AE_new' or mode=='AE_Platon' or mode =='Swap_LIA' or mode =='Swap_LIA_warmup'\
            or mode == 'MFIM_nomaps_warmup' or mode == 'pSp_Chain':
        E = dnnlib.util.construct_class_by_name(**E_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        if mode == 'MFIM' or mode == 'AE' or mode == 'AE_new' or mode=='AE_Platon' or mode == 'MFIM_nomaps_warmup' or mode == 'pSp_Chain':
            E = torch.nn.SyncBatchNorm.convert_sync_batchnorm(E)
    elif mode =='single_id'or mode=='opt':
        pass
        E_exp = ExpMappingNet(coeff_nc=50, descriptor_nc=512, num_layers=3, residual_layers=[1]).train().requires_grad_(False).to(device)
        G.w_avg = G.backbone.mapping.w_avg.unsqueeze(0).unsqueeze(1).repeat(1,14,1).detach().cpu().requires_grad_(False).to(device)
        #w_opt = torch.nn.Parameter(torch.zeros((1,14,512)), requires_grad=False).to(device)
        #G.w_avg.requires_grad = True
        #print(G.w_avg[0][0],'sdsD!!')
        #exit()
        
    # Load pretrained smthng for finetuning or inversion
    if (pretrain is not None) and (rank == 0):
    #if True:
        print(f'Load pre-trained weight from "{pretrain}"')
        with dnnlib.util.open_url(pretrain) as f:
            pretrained_data = legacy.load_network_pkl(f)
        # Load pre-trained Generator
        misc.copy_params_and_buffers(pretrained_data['G_ema'], G, require_all=False)
        # Load pre-trained Discriminator
        misc.copy_params_and_buffers(pretrained_data['D'], D, require_all=False)
   

    

    # Resume from existing pickle.
    #if (resume_pkl is not None) and (rank == 0):
    if False:
        resume_pkl='/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl'
        print(f'Resuming from "{resume_pkl}"')
        ep ='/home/nas2_userF/gyojunggu/gyojung/faceswap/eg3d/eg3d/ae_new_128_w+_1202_noid_lock_eg3d/00003-ffhq-FFHQ_png_512-gpus3-batch12-gamma1/E-snapshot-000501.pth'
        gp = ep.replace('E-snap','G-snap')
        #dp = ep.replace('E-snap','D-snap')

        E_ckpt = torch.load(ep)
        #G_ckpt = torch.load(gp)
        #D_ckpt = torch.load(dp)

        E.load_state_dict(E_ckpt)
        #G.load_state_dict(G_ckpt)
        with dnnlib.util.open_url(resume_pkl) as f:
            pretrained_data = legacy.load_network_pkl(f)
        misc.copy_params_and_buffers(pretrained_data['G_ema'], G, require_all=False)
        # Load pre-trained Discriminator
        misc.copy_params_and_buffers(pretrained_data['D'], D, require_all=False)
        #D.load_state_dict(D_ckpt)

        '''
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        '''
    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        #c = torch.empty([batch_gpu, G.c_dim], device=device)
        #import pdb;pdb.set_trace()
        #img = misc.print_module_summary(G, [z, c])
        #misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if mode == 'single_id':
        module_list = [E_exp, G, G.w_avg] #G.w_avg]
    elif mode == 'opt':
        module_list = [G]
    elif mode == 'EG3D':
        module_list = [G, D, G_ema, augment_pipe]
    elif mode == 'AE' or mode == 'AE_new' or mode=='AE_Platon' or mode=='AE_new_Platon' or mode == 'pSp_Chain':
        module_list = [E, G]
        if 'gan' in loss_selection:
            module_list.append(D)
    elif mode == 'Swap_LIA' or mode == 'Swap_LIA_warmup' or mode == 'MFIM_nomaps_warmup' or mode=='MFIM':
        module_list = [E, G]
        if 'gan' in loss_selection:
            module_list.append(D)
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    #for module in [G, D, G_ema, augment_pipe]:
    for module in module_list:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    if mode =='MFIM' or mode == 'AE' or mode == 'AE_new' or mode=='AE_Platon' or mode =='AE_new_Platon' or mode =='Swap_LIA' or mode =='Swap_LIA_warmup' \
            or mode=='MFIM_nomaps_warmup' or mode == 'pSp_Chain':
        loss = dnnlib.util.construct_class_by_name(device=device, rank=rank, G=G, D=D, E=E, augment_pipe=augment_pipe,\
            loss_selection = loss_selection, invert_map = invert_map, resolution_encode=resolution_encode, mode=mode, start_from_avg = start_from_avg, **loss_kwargs) # subclass of training.loss.Loss
    elif mode == 'single_id':
        
        src_path ='/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/inthewild_data_img/candal.png'
        src_img = np.array(PIL.Image.open(src_path)).transpose(2,0,1)
        src_ldmk = np.load(src_path.replace('img','kps').replace('.png','.png.npy'),allow_pickle=True)
        src_img = (torch.tensor(src_img).to(rank).to(torch.float32) / 127.5 - 1)

        src_ldmk = torch.tensor(src_ldmk).to(rank)

        loss = dnnlib.util.construct_class_by_name(device=device, rank=rank, G=G, D=D, E=E_exp, augment_pipe=augment_pipe,\
            loss_selection = loss_selection, invert_map = invert_map, resolution_encode=resolution_encode, mode=mode, \
                 start_from_avg = start_from_avg, src_img=src_img,src_ldmk=src_ldmk,**loss_kwargs) # subclass of training.loss.Loss
    elif mode == 'opt':
        
        src_path ='/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/inthewild_data_img/candal.png'
        src_img = np.array(PIL.Image.open(src_path)).transpose(2,0,1)
        src_ldmk = np.load(src_path.replace('img','kps').replace('.png','.png.npy'),allow_pickle=True)
        src_img = (torch.tensor(src_img).to(rank).to(torch.float32) / 127.5 - 1)

        src_ldmk = torch.tensor(src_ldmk).to(rank)

        loss = dnnlib.util.construct_class_by_name(device=device, rank=rank, G=G, D=D, E=None, augment_pipe=augment_pipe,\
            loss_selection = loss_selection, invert_map = invert_map, resolution_encode=resolution_encode, mode=mode, \
                 start_from_avg = start_from_avg, src_img=src_img,src_ldmk=src_ldmk,**loss_kwargs) # subclass of training.loss.Loss
    else:
        loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    phases = []
  
      
    if mode == 'single_id':
        params = []
        params += list(E_exp.parameters())
        params.append(G.w_avg)
        
        opt = dnnlib.util.construct_class_by_name(params=params, **E_opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name='Gmain', module=[E_exp, G], opt=opt, interval=1)]
    elif mode == 'opt':
        params = []
        #params += list(E_exp.parameters())
        params.append(G.w_avg)
        
        opt = dnnlib.util.construct_class_by_name(params=params, **E_opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name='Gmain', module=[G], opt=opt, interval=1)]
    elif mode == 'AE' or mode == 'pSp_Chain':
        #import pdb;pdb.set_trace()
        params = []
        params += list(E.parameters())
        
        opt = dnnlib.util.construct_class_by_name(params=params, **E_opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name='Gmain', module=[E,G], opt=opt, interval=1)]

    elif mode == 'AE_new':
        #import pdb;pdb.set_trace()
        params = []
        params += list(E.parameters())
        params += list(G.backbone.parameters())
        
        opt = dnnlib.util.construct_class_by_name(params=params, **E_opt_kwargs) # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name='Gmain', module=[E,G], opt=opt, interval=1)]

    elif mode=='AE_Platon' or mode =='AE_new_Platon':
        #import pdb;pdb.set_trace()
        params = []
        params += list(E.parameters())
        params += list(G.backbone.parameters())
        
        opt = dnnlib.util.construct_class_by_name(params=params, **E_opt_kwargs) # subclass of torch.optim.Optimizer
        #phases += [dnnlib.EasyDict(name='Gmain', module=[E,G], opt=opt, interval=1)]
        if 'gan' in loss_selection:
            opt_list = [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]
        else:
            opt_list = [('G', G, G_opt_kwargs, G_reg_interval)]
        for name, module, opt_kwargs, reg_interval in opt_list:
            
            
            
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]

            if name == 'G':
                opt = dnnlib.util.construct_class_by_name(module.backbone.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            else:
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
           
            if (name == 'G') and ('tv' in loss_selection):
                phases += [dnnlib.EasyDict(name=name+'reg', module=[module], opt=opt, interval=reg_interval)]

            phases += [dnnlib.EasyDict(name=name+'main', module=[module], opt=opt, interval=1)]

            if mode=='AE_new_Platon':
                phases += [dnnlib.EasyDict(name='Daux', module=[D], opt=D_opt_kwargs, interval=D_reg_interval)]

    elif mode=='Swap_LIA' or mode=='Swap_LIA_warmup' or mode == 'MFIM_nomaps_warmup' or mode == 'MFIM':
        #import pdb;pdb.set_trace()
        #params = []
        #params += list(E.parameters())
        #params += list(G.backbone.parameters())
        
        #opt = dnnlib.util.construct_class_by_name(params=params, **E_opt_kwargs) # subclass of torch.optim.Optimizer
        #phases += [dnnlib.EasyDict(name='Gmain', module=[E,G], opt=opt, interval=1)]
        if 'gan' not in loss_selection:

            opt = dnnlib.util.construct_class_by_name(params = E.parameters(), **E_opt_kwargs)
            phases = [dnnlib.EasyDict(name='Gmain', module=[E,G], opt=opt, interval=1)] 

        elif 'gan' in loss_selection:
            opt_list = [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]
        
            for name, module, opt_kwargs, reg_interval in opt_list:
                
                
                
                

                if name == 'G':

                    G_mb_ratio = G_reg_interval / (G_reg_interval + 1)
                    G_opt_kwargs = dnnlib.EasyDict(G_opt_kwargs)
                    G_opt_kwargs.lr = G_opt_kwargs.lr * G_mb_ratio
                    G_opt_kwargs.betas = [beta ** G_mb_ratio for beta in G_opt_kwargs.betas]    

                    EG_params = []
                    EG_params += list(E.parameters())
                    EG_params += list(G.backbone.parameters())

                    opt_EG = dnnlib.util.construct_class_by_name(EG_params, **G_opt_kwargs) # subclass of torch.optim.Optimizer

                    phases += [dnnlib.EasyDict(name='Gmain', module=[E,G], opt=opt_EG, interval=1)]
                    if 'gan_aux' in loss_selection:
                        phases += [dnnlib.EasyDict(name='Gaux', module=[E,G], opt=opt_EG, interval=4)]

                    if 'tv' in loss_selection:
                        phases += [dnnlib.EasyDict(name='Greg', module=[E,G], opt=opt_EG, interval=G_reg_interval)]
               
                elif name == 'D':

                    D_mb_ratio = D_reg_interval / (D_reg_interval + 1)
                    D_opt_kwargs = dnnlib.EasyDict(D_opt_kwargs)
                    D_opt_kwargs.lr = D_opt_kwargs.lr * D_mb_ratio
                    D_opt_kwargs.betas = [beta ** D_mb_ratio for beta in D_opt_kwargs.betas]  
                    
                    opt_D = dnnlib.util.construct_class_by_name(D.parameters(), **D_opt_kwargs) # subclass of torch.optim.Optimizer

                    phases += [dnnlib.EasyDict(name='Dmain', module=[D], opt=opt_D, interval=1)]
                    if 'gan_aux' in loss_selection:
                        phases += [dnnlib.EasyDict(name='Daux', module=[D], opt=opt_D, interval=4)]
                    phases += [dnnlib.EasyDict(name='Dreg', module=[D], opt=opt_D, interval=D_reg_interval)]
                
                    
                
              
                
                    
        else:
            
            opt = dnnlib.util.construct_class_by_name(params=E.parameters(), **E_opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name='Gmain', module=[E,G], opt=opt, interval=1)]

    else:
        for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
            if reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
            else: # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        print(f'{phase.name} START !!')
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)
    #breakpoint()
    # Export sample images.
    '''
    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
    '''

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            if mode == 'single_id' or mode == 'opt':
                phase_real_frm,phase_real_msk, phase_real_ldmk, phase_real_c = next(training_set_iterator)
                phase_real_frm = (phase_real_frm.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                #phase_real_msk = (phase_real_msk.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_msk = (phase_real_msk.to(device).to(torch.float32) / 255).split(batch_gpu) #XXX squashing to 0-1
                phase_real_ldmk = phase_real_ldmk.to(device).split(batch_gpu)
                #phase_real_c = phase_real_c.to(device).split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)

            elif mode == 'AE' or mode=='AE_new' or mode=='AE_Platon' or mode=='AE_new_Platon' or mode == 'pSp_Chain':

                phase_real_img, phase_real_c = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)

            elif mode=='Swap_LIA' or mode == 'Swap_LIA_warmup' or mode == 'MFIM_nomaps_warmup' or mode == 'MFIM':
                phase_real_img_src, phase_real_img_tgt, phase_real_c = next(training_set_iterator)

                phase_real_img_src = (phase_real_img_src.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_img_tgt = (phase_real_img_tgt.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)

                phase_real_c = phase_real_c.to(device).split(batch_gpu)

            else:
                phase_real_img, phase_real_c = next(training_set_iterator)
                phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
                phase_real_c = phase_real_c.to(device).split(batch_gpu)
                all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
                all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
                all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
                all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
                all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        if mode == 'single_id' or mode =='opt':

            for phase in phases:

                if batch_idx % phase.interval != 0:
                    continue

                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))

                # Accumulate gradients.
                

                phase.opt.zero_grad(set_to_none=True)

                for module in phase.module:
                    module.requires_grad_(True)

                for real_frm, real_c, real_msk, real_ldmk in zip(phase_real_frm, phase_real_c, phase_real_msk, phase_real_ldmk):
                    
                    if phase.name == 'Gmain':
                        real_img_save, gen_img_save, interm_output = \
                            loss.accumulate_gradients(phase=phase.name, frm=real_frm, c=real_c, msk=real_msk, ldmk=real_ldmk, w_opt=None, gain=phase.interval, cur_nimg=cur_nimg)
                    
                
                   
                for module in phase.module:
                    module.requires_grad_(False)
                

                # Update weights.
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    params = []
                    for single_module in phase.module:
                        try:
                            params += [param for param in single_module.parameters() if param.numel() > 0 and param.grad is not None]
                        except:
                            params += [param for param in single_module if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad in zip(params, grads):
                            param.grad = grad.reshape(param.shape)
                    phase.opt.step()
                    #if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                    #    if mode=='AE':
                    #        print('Scheduler step!')

                # Phase done.
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

            # Update G_ema.
            '''
            with torch.autograd.profiler.record_function('Gema'):
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)
                G_ema.neural_rendering_resolution = G.neural_rendering_resolution
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()
            '''
            # Update state.
            cur_nimg += batch_size
            batch_idx += 1

           
           
            # Perform maintenance tasks once per tick.
            done = (cur_nimg >= total_kimg * 1000)
            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            #if False:
                if rank==0:
                    if cur_nimg%1000 == 0:
                        print(cur_nimg, " : cur_nimg")
                    #print(mode)
                continue
            
            # Print status line, accumulating the same information in training_stats.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
            training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
            if rank == 0:
                print(' '.join(fields))

            # Check for abort.
            if (not done) and (abort_fn is not None) and abort_fn():
                done = True
                if rank == 0:
                    print()
                    print('Aborting...')

            # Save image snapshot.
            if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                
                
                print('Saving Images!!')
                images = torch.cat([real_img_save['frm'], real_img_save['frm']*real_img_save['msk'], gen_img_save['image'].detach()], 0).cpu().numpy()
                images_depth = gen_img_save['image_depth'].detach().cpu().numpy()
                #print(images.shape, images_depth.shape)
                #breakpoint()
                aligned = torch.cat([interm_output['rec_aligned'].detach(),interm_output['src_aligned'].repeat(images.shape[0],1,1,1)],0).cpu().numpy()
                save_image_grid(images, os.path.join(run_dir, f'{phase.name}_fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=(int(batch_size/num_gpus),3))
                save_image_grid(images_depth, os.path.join(run_dir, f'{phase.name}_fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=(int(batch_size/num_gpus),1))
                save_image_grid(aligned, os.path.join(run_dir, f'{phase.name}_fakes{cur_nimg//1000:06d}_aligned.png'), drange=[-1,1], grid_size=(int(batch_size/num_gpus),4))
               
               
            # Save network snapshot.
            snapshot_pkl = None
            snapshot_data = None
            #(network_snapshot_ticks*10)
            if (network_snapshot_ticks is not None) and (done or cur_tick % (network_snapshot_ticks*5) == 0):
                
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))   
                if mode == 'opt':
                    check_modules = [('G', G)]
                else:
                    check_modules = [('G', G), ('E', E_exp)]

                if 'gan' in loss_selection:
                    check_modules.append(('D', D))
    
                for name, module in check_modules:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module # conserve memory
                
                snapshot_ckpt_G = os.path.join(run_dir, f'G-snapshot-{cur_nimg//1000:06d}.pth')
                snapshot_ckpt_E = os.path.join(run_dir, f'E-snapshot-{cur_nimg//1000:06d}.pth')

                if 'gan' in loss_selection:
                    snapshot_ckpt_D = os.path.join(run_dir, f'D-snapshot-{cur_nimg//1000:06d}.pth')

                
                if rank == 0:
                    torch.save(snapshot_data['G'].state_dict(), snapshot_ckpt_G)
                    if mode=='single_id':
                        torch.save(snapshot_data['E'].state_dict(), snapshot_ckpt_E)

                    if 'gan' in loss_selection:
                        torch.save(snapshot_data['D'].state_dict(), snapshot_ckpt_D)

                    print('***'*10)
                    print('pickle dumpped!')
                    print(f'cur tick: {cur_tick}')
                    print(f'cur nimg: {cur_nimg}')
                    print('***'*10)

                    #with open(snapshot_pkl, 'wb') as f:
                    #    pickle.dump(snapshot_data, f)

            # Evaluate metrics.
            '''
            if (snapshot_data is not None) and (len(metrics) > 0):
                if rank == 0:
                    print(run_dir)
                    print('Evaluating metrics...')
                for metric in metrics:
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)
            del snapshot_data # conserve memory
            '''
            # Collect statistics.
            for phase in phases:
                value = []
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
            stats_collector.update()
            stats_dict = stats_collector.as_dict()

            # Update logs.
            timestamp = time.time()
            if stats_jsonl is not None:
                fields = dict(stats_dict, timestamp=timestamp)
                stats_jsonl.write(json.dumps(fields) + '\n')
                stats_jsonl.flush()
            if stats_tfevents is not None:
                global_step = int(cur_nimg / 1e3)
                walltime = timestamp - start_time
                for name, value in stats_dict.items():
                    stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
                for name, value in stats_metrics.items():
                    stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                stats_tfevents.flush()
            if progress_fn is not None:
                progress_fn(cur_nimg // 1000, total_kimg)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

            if done:
                break
        elif mode == 'AE' or mode=='AE_new' or mode=='AE_Platon' or mode=='AE_new_Platon' or mode == 'pSp_Chain':

            for phase in phases:

                if batch_idx % phase.interval != 0:
                    continue

                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))

                # Accumulate gradients.
                

                phase.opt.zero_grad(set_to_none=True)

                for module in phase.module:
                    module.requires_grad_(True)

                for real_img, real_c in zip(phase_real_img, phase_real_c):
                    
                    if phase.name == 'Gmain':
                        real_img_save, gen_img_save = \
                            loss.accumulate_gradients(phase=phase.name, img_src=real_img, c=real_c, gain=phase.interval, cur_nimg=cur_nimg)
                    
                    if phase.name == 'Gaux':
                        real_img_save, gen_img_save = \
                            loss.accumulate_gradients(phase=phase.name, img_src=real_img, c=real_c, gain=phase.interval, cur_nimg=cur_nimg)
                    
                    else:
                        loss.accumulate_gradients(phase=phase.name, img_src=real_img, c=real_c, gain=phase.interval, cur_nimg=cur_nimg)
                   
                for module in phase.module:
                    module.requires_grad_(False)
                

                # Update weights.
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    params = []
                    for single_module in phase.module:
                        params += [param for param in single_module.parameters() if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad in zip(params, grads):
                            param.grad = grad.reshape(param.shape)
                    phase.opt.step()
                    #if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                    #    if mode=='AE':
                    #        print('Scheduler step!')

                # Phase done.
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

            # Update G_ema.
            '''
            with torch.autograd.profiler.record_function('Gema'):
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)
                G_ema.neural_rendering_resolution = G.neural_rendering_resolution
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()
            '''
            # Update state.
            cur_nimg += batch_size
            batch_idx += 1

           
           
            # Perform maintenance tasks once per tick.
            done = (cur_nimg >= total_kimg * 1000)
            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            #if False:
                if rank==0:
                    if cur_nimg%1000 == 0:
                        print(cur_nimg, " : cur_nimg")
                    #print(mode)
                continue

            # Print status line, accumulating the same information in training_stats.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
            training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
            if rank == 0:
                print(' '.join(fields))

            # Check for abort.
            if (not done) and (abort_fn is not None) and abort_fn():
                done = True
                if rank == 0:
                    print()
                    print('Aborting...')

            # Save image snapshot.
            if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                
                if mode == 'AE_Platon' or mode=='AE_new_Platon':

                    images = torch.cat((real_img_save['image_src'], gen_img_save['src']['image'].detach(), gen_img_save['roll']['image'].detach()), 0).cpu().numpy()
                    images_depth = torch.cat((gen_img_save['src']['image_depth'].detach(),gen_img_save['roll']['image_depth'].detach()),0).cpu().numpy()
                    #print(images.shape, images_depth.shape)
                    save_image_grid(images, os.path.join(run_dir, f'{phase.name}_fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=(int(batch_size/num_gpus),3))
                    save_image_grid(images_depth, os.path.join(run_dir, f'{phase.name}_fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=(int(batch_size/num_gpus),2))
               
                else:

                    images = torch.cat((real_img_save['image_src'], gen_img_save['image'].detach()), 0).cpu().numpy()
                    images_depth = gen_img_save['image_depth'].detach().cpu().numpy()

                    save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=(int(batch_size/num_gpus),2))
                    save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=(int(batch_size/num_gpus),1))

               
               
            # Save network snapshot.
            snapshot_pkl = None
            snapshot_data = None
            #(network_snapshot_ticks*10)
            if (network_snapshot_ticks is not None) and (done or cur_tick % (network_snapshot_ticks*5) == 0):
                
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))   
                
                check_modules = [('G', G), ('E', E)]

                if 'gan' in loss_selection:
                    check_modules.append(('D', D))
    
                for name, module in check_modules:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module # conserve memory
                
                snapshot_ckpt_G = os.path.join(run_dir, f'G-snapshot-{cur_nimg//1000:06d}.pth')
                snapshot_ckpt_E = os.path.join(run_dir, f'E-snapshot-{cur_nimg//1000:06d}.pth')

                if 'gan' in loss_selection:
                    snapshot_ckpt_D = os.path.join(run_dir, f'D-snapshot-{cur_nimg//1000:06d}.pth')

                
                if rank == 0:
                    torch.save(snapshot_data['G'].state_dict(), snapshot_ckpt_G)
                    torch.save(snapshot_data['E'].state_dict(), snapshot_ckpt_E)

                    if 'gan' in loss_selection:
                        torch.save(snapshot_data['D'].state_dict(), snapshot_ckpt_D)

                    print('***'*10)
                    print('pickle dumpped!')
                    print(f'cur tick: {cur_tick}')
                    print(f'cur nimg: {cur_nimg}')
                    print('***'*10)

                    #with open(snapshot_pkl, 'wb') as f:
                    #    pickle.dump(snapshot_data, f)

            # Evaluate metrics.
            '''
            if (snapshot_data is not None) and (len(metrics) > 0):
                if rank == 0:
                    print(run_dir)
                    print('Evaluating metrics...')
                for metric in metrics:
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)
            del snapshot_data # conserve memory
            '''
            # Collect statistics.
            for phase in phases:
                value = []
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
            stats_collector.update()
            stats_dict = stats_collector.as_dict()

            # Update logs.
            timestamp = time.time()
            if stats_jsonl is not None:
                fields = dict(stats_dict, timestamp=timestamp)
                stats_jsonl.write(json.dumps(fields) + '\n')
                stats_jsonl.flush()
            if stats_tfevents is not None:
                global_step = int(cur_nimg / 1e3)
                walltime = timestamp - start_time
                for name, value in stats_dict.items():
                    stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
                for name, value in stats_metrics.items():
                    stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                stats_tfevents.flush()
            if progress_fn is not None:
                progress_fn(cur_nimg // 1000, total_kimg)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

            if done:
                break
            
        elif mode == 'Swap_LIA' or mode == 'Swap_LIA_warmup' or mode == 'MFIM_nomaps_warmup' or mode=='MFIM':
    
            
            for phase in phases:

                if batch_idx % phase.interval != 0:
                    continue

                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))

                # Accumulate gradients.
                

                phase.opt.zero_grad(set_to_none=True)

                for module in phase.module:
                    module.requires_grad_(True)

                for real_img_src, real_img_tgt, real_c in zip(phase_real_img_src, phase_real_img_tgt, phase_real_c):
                    
                    if phase.name == 'Gmain':
                        real_img_save, gen_img_save = \
                            loss.accumulate_gradients(phase=phase.name, img_src=real_img_src, img_tgt=real_img_tgt, c=real_c, gain=phase.interval, cur_nimg=cur_nimg)
                        save_flag = 'Gmain'
                    elif phase.name == 'Gaux':
                        real_img_save, gen_img_save = \
                            loss.accumulate_gradients(phase=phase.name, img_src=real_img_src, img_tgt=real_img_tgt, c=real_c, gain=phase.interval, cur_nimg=cur_nimg)
                        save_flag = 'Gaux'
                    else:
                        loss.accumulate_gradients(phase=phase.name, img_src=real_img_src, img_tgt=real_img_tgt, c=real_c, gain=phase.interval, cur_nimg=cur_nimg)
                   
                for module in phase.module:
                    module.requires_grad_(False)
                

                # Update weights.
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    params = []
                    for single_module in phase.module:
                        params += [param for param in single_module.parameters() if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad in zip(params, grads):
                            param.grad = grad.reshape(param.shape)
                    phase.opt.step()
                    #if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                    #    if mode=='AE':
                    #        print('Scheduler step!')

                # Phase done.
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

            # Update G_ema.
            '''
            with torch.autograd.profiler.record_function('Gema'):
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)
                G_ema.neural_rendering_resolution = G.neural_rendering_resolution
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()
            '''
            # Update state.
            cur_nimg += batch_size
            batch_idx += 1

           
           
            # Perform maintenance tasks once per tick.
            done = (cur_nimg >= total_kimg * 1000)
            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            #if False:
                if rank==0:
                    if cur_nimg%1000 == 0:
                        print(cur_nimg, " : cur_nimg")
                    #print(mode)
                continue

            # Print status line, accumulating the same information in training_stats.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
            training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
            if rank == 0:
                print(' '.join(fields))

            # Check for abort.
            if (not done) and (abort_fn is not None) and abort_fn():
                done = True
                if rank == 0:
                    print()
                    print('Aborting...')

            # Save image snapshot.
            if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                
                if mode == 'Swap_LIA' or mode =='MFIM_nomaps_warmup' or mode == 'MFIM':
                    if True:
                        print(f'Try to save imgs from {save_flag}')
                        images = torch.cat((real_img_save['image_src'], real_img_save['image_tgt'], gen_img_save['image'].detach()), 0).cpu().numpy()
                        images_depth = gen_img_save['image_depth'].detach().cpu().numpy()

                        save_image_grid(images, os.path.join(run_dir, f'{save_flag}_fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=(int(batch_size/num_gpus),3))
                        save_image_grid(images_depth, os.path.join(run_dir, f'{save_flag}_fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=(int(batch_size/num_gpus),1))
                elif mode == 'Swap_LIA_warmup':
                    if phase.name == 'Gaux' or phase.name == 'Gmain':
                        images = torch.cat((real_img_save['image_tgt'], gen_img_save['image'].detach()), 0).cpu().numpy()
                        images_depth = gen_img_save['image_depth'].detach().cpu().numpy()

                        save_image_grid(images, os.path.join(run_dir, f'{phase.name}_fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=(int(batch_size/num_gpus),2))
                        save_image_grid(images_depth, os.path.join(run_dir, f'{phase.name}_fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=(int(batch_size/num_gpus),1))
           
                else:
                    print('FUCKEDUP')
                    exit()
               
            # Save network snapshot.
            snapshot_pkl = None
            snapshot_data = None
            #(network_snapshot_ticks*10)
            if (network_snapshot_ticks is not None) and (done or cur_tick % (network_snapshot_ticks*5) == 0):
                
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))   
                
                check_modules = [('G', G), ('E', E)]

                if 'gan' in loss_selection:
                    check_modules.append(('D', D))
    
                for name, module in check_modules:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module # conserve memory
                
                snapshot_ckpt_G = os.path.join(run_dir, f'G-snapshot-{cur_nimg//1000:06d}.pth')
                snapshot_ckpt_E = os.path.join(run_dir, f'E-snapshot-{cur_nimg//1000:06d}.pth')

                if 'gan' in loss_selection:
                    snapshot_ckpt_D = os.path.join(run_dir, f'D-snapshot-{cur_nimg//1000:06d}.pth')

                
                if rank == 0:
                    torch.save(snapshot_data['G'].state_dict(), snapshot_ckpt_G)
                    torch.save(snapshot_data['E'].state_dict(), snapshot_ckpt_E)

                    if 'gan' in loss_selection:
                        torch.save(snapshot_data['D'].state_dict(), snapshot_ckpt_D)

                    print('***'*10)
                    print('pickle dumpped!')
                    print(f'cur tick: {cur_tick}')
                    print(f'cur nimg: {cur_nimg}')
                    print('***'*10)

                    #with open(snapshot_pkl, 'wb') as f:
                    #    pickle.dump(snapshot_data, f)

            # Evaluate metrics.
            '''
            if (snapshot_data is not None) and (len(metrics) > 0):
                if rank == 0:
                    print(run_dir)
                    print('Evaluating metrics...')
                for metric in metrics:
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)
            del snapshot_data # conserve memory
            '''
            # Collect statistics.
            for phase in phases:
                value = []
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
            stats_collector.update()
            stats_dict = stats_collector.as_dict()

            # Update logs.
            timestamp = time.time()
            if stats_jsonl is not None:
                fields = dict(stats_dict, timestamp=timestamp)
                stats_jsonl.write(json.dumps(fields) + '\n')
                stats_jsonl.flush()
            if stats_tfevents is not None:
                global_step = int(cur_nimg / 1e3)
                walltime = timestamp - start_time
                for name, value in stats_dict.items():
                    stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
                for name, value in stats_metrics.items():
                    stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                stats_tfevents.flush()
            if progress_fn is not None:
                progress_fn(cur_nimg // 1000, total_kimg)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time

            if done:
                break



    #----------------------------------------------------------------------------

        else:
            for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
                if batch_idx % phase.interval != 0:
                    continue
                if phase.start_event is not None:
                    phase.start_event.record(torch.cuda.current_stream(device))

                # Accumulate gradients.
                phase.opt.zero_grad(set_to_none=True)
                phase.module.requires_grad_(True)
                for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
                phase.module.requires_grad_(False)

                # Update weights.
                with torch.autograd.profiler.record_function(phase.name + '_opt'):
                    params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                    if len(params) > 0:
                        flat = torch.cat([param.grad.flatten() for param in params])
                        if num_gpus > 1:
                            torch.distributed.all_reduce(flat)
                            flat /= num_gpus
                        misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                        grads = flat.split([param.numel() for param in params])
                        for param, grad in zip(params, grads):
                            param.grad = grad.reshape(param.shape)
                    phase.opt.step()

                # Phase done.
                if phase.end_event is not None:
                    phase.end_event.record(torch.cuda.current_stream(device))

            # Update G_ema.
            with torch.autograd.profiler.record_function('Gema'):
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)
                G_ema.neural_rendering_resolution = G.neural_rendering_resolution
                G_ema.rendering_kwargs = G.rendering_kwargs.copy()

            # Update state.
            cur_nimg += batch_size
            batch_idx += 1

            # Execute ADA heuristic.
            if (ada_stats is not None) and (batch_idx % ada_interval == 0):
                ada_stats.update()
                adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
                augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

            # Perform maintenance tasks once per tick.
            done = (cur_nimg >= total_kimg * 1000)
            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
                continue

            # Print status line, accumulating the same information in training_stats.
            tick_end_time = time.time()
            fields = []
            fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
            fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
            fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
            fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
            fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
            fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
            fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()
            fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
            training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
            training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
            if rank == 0:
                print(' '.join(fields))

            # Check for abort.
            if (not done) and (abort_fn is not None) and abort_fn():
                done = True
                if rank == 0:
                    print()
                    print('Aborting...')

            # Save image snapshot.
            if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
                out = [G_ema(z=z, c=c, noise_mode='const') for z, c in zip(grid_z, grid_c)]
                images = torch.cat([o['image'].cpu() for o in out]).numpy()
                images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
                images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
                save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw.png'), drange=[-1,1], grid_size=grid_size)
                save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

                #--------------------
                # # Log forward-conditioned images

                # forward_cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
                # intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                # forward_label = torch.cat([forward_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                # grid_ws = [G_ema.mapping(z, forward_label.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
                # out = [G_ema.synthesis(ws, c=c, noise_mode='const') for ws, c in zip(grid_ws, grid_c)]

                # images = torch.cat([o['image'].cpu() for o in out]).numpy()
                # images_raw = torch.cat([o['image_raw'].cpu() for o in out]).numpy()
                # images_depth = -torch.cat([o['image_depth'].cpu() for o in out]).numpy()
                # save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_f.png'), drange=[-1,1], grid_size=grid_size)
                # save_image_grid(images_raw, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_raw_f.png'), drange=[-1,1], grid_size=grid_size)
                # save_image_grid(images_depth, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_depth_f.png'), drange=[images_depth.min(), images_depth.max()], grid_size=grid_size)

                #--------------------
                # # Log Cross sections

                # grid_ws = [G_ema.mapping(z, c.expand(z.shape[0], -1)) for z, c in zip(grid_z, grid_c)]
                # out = [sample_cross_section(G_ema, ws, w=G.rendering_kwargs['box_warp']) for ws, c in zip(grid_ws, grid_c)]
                # crossections = torch.cat([o.cpu() for o in out]).numpy()
                # save_image_grid(crossections, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_crossection.png'), drange=[-50,100], grid_size=grid_size)

            # Save network snapshot.
            snapshot_pkl = None
            snapshot_data = None
            if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
                snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
                for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                    del module # conserve memory
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                if rank == 0:
                    with open(snapshot_pkl, 'wb') as f:
                        pickle.dump(snapshot_data, f)

            # Evaluate metrics.
            if (snapshot_data is not None) and (len(metrics) > 0):
                if rank == 0:
                    print(run_dir)
                    print('Evaluating metrics...')
                for metric in metrics:
                    result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                        dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                    stats_metrics.update(result_dict.results)
            del snapshot_data # conserve memory

            # Collect statistics.
            for phase in phases:
                value = []
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
            stats_collector.update()
            stats_dict = stats_collector.as_dict()

            # Update logs.
            timestamp = time.time()
            if stats_jsonl is not None:
                fields = dict(stats_dict, timestamp=timestamp)
                stats_jsonl.write(json.dumps(fields) + '\n')
                stats_jsonl.flush()
            if stats_tfevents is not None:
                global_step = int(cur_nimg / 1e3)
                walltime = timestamp - start_time
                for name, value in stats_dict.items():
                    stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
                for name, value in stats_metrics.items():
                    stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                stats_tfevents.flush()
            if progress_fn is not None:
                progress_fn(cur_nimg // 1000, total_kimg)

            # Update state.
            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
            if done:
                break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

    #----------------------------------------------------------------------------
