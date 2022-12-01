# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import json


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator

#XXX python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=0-3 --network=/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl
#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------
def init_dataset_kwargs(data, mode):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False) #mode=mode)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
#----------------------------------------------------------------------------
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    

    E_path = '/home/nas2_userF/gyojunggu/gyojung/faceswap/eg3d/eg3d/ae_new_128_w+_1129/00007-ffhq-FFHQ_png_512-gpus2-batch4-gamma1/E-snapshot-000100.pth'
    G_path = '/home/nas2_userF/gyojunggu/gyojung/faceswap/eg3d/eg3d/ae_new_128_w+_1129/00007-ffhq-FFHQ_png_512-gpus2-batch4-gamma1/G-snapshot-000100.pth'
    E_ckpt = torch.load(E_path)
    G_ckpt = torch.load(G_path)
    train_opt_path = '/home/nas2_userF/gyojunggu/gyojung/faceswap/eg3d/eg3d/ae_new_128_w+_1129/00007-ffhq-FFHQ_png_512-gpus2-batch4-gamma1/training_options.json'
    with open(train_opt_path) as js:
        opt_json = json.load(js)
    common_kwargs = dict(c_dim=25, img_resolution=512, img_channels=96)
    #breakpoint()
    del G_ckpt['dataset_label_std']
    G = dnnlib.util.construct_class_by_name(**opt_json['G_kwargs'], **common_kwargs).eval().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G.load_state_dict(G_ckpt)
    import sys
    sys.path.append('./training/pixel2style2pixel')
    sys.path.append('./training/DECA')
    E_kwargs = dnnlib.EasyDict(class_name='training.pixel2style2pixel.models.encoders.psp_encoders.GradualStyleEncoder',\
                img_res=256, block_selections = None, w_type = 'w+')
    E = dnnlib.util.construct_class_by_name(**opt_json['E_kwargs']).eval().requires_grad_(False).to(device)
    E.load_state_dict(E_ckpt)
    #G_new = TriPlaneGenerator(c_dim =25, img_channels=96, img_resolution=512, **opt_json['G_kwargs']).eval().requires_grad_(False).to(device)
        
    #breakpoint()
    #with dnnlib.util.open_url(network_pkl) as f:
        
    #    G = legacy.load_network_pkl(f)['G'].to(device) # type: ignore
    #    E = legacy.load_network_pkl(f)['E'].to(device)
    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    '''
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
        E_kwargs = dnnlib.EasyDict(class_name='training.pixel2style2pixel.models.encoders.psp_encoders.GradualStyleEncoder',\
                img_res=256, block_selections = None, w_type = 'w+')
        E_new = dnnlib.util.construct_class_by_name(**E_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(E, E_new, require_all=True)
    '''
    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    #XXX dataload
    data_path = '/home/nas2_userG/junhahyung/FFHQ_png_512.zip'
    training_set_kwargs, dataset_name = init_dataset_kwargs(data=data_path, mode=None)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=0, num_replicas=1, seed=seeds[0])
    data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=1, **data_loader_kwargs))
    
    #XXX
    from torch_utils.ops import upfirdn2d
    from training.dual_discriminator import filtered_resizing
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        #z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        real_img, real_c = next(training_set_iterator)
        real_img = (real_img.to(device).to(torch.float32) / 127.5 - 1)
        
        resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        
        real_img= filtered_resizing(real_img, size=256, f=resample_filter, filter_mode='antialiased')
        real_c = real_c.to(device)
        imgs = []
        angle_p = -0.2
        ws = E(real_img)
        for angle_y, angle_p in [(1.2, angle_p), (0, angle_p), (-1.2, angle_p)]:
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            #ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            #import pdb;pdb.set_trace()
            
            img = G.synthesis(ws, None, camera_params)['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)

        img = torch.cat(imgs, dim=2)

        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1
            
            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            breakpoint()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
