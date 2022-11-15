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


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


#----------------------------------------------------------------------------
#XXX python gen_samples.py --outdir=out_extreme --trunc=0.3 --shapes=False --seeds=0-3 --network /home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl
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
def histogram_equalizer(temp_l1):
    temp_l1 = torch.sum(temp_l1,dim=(0,1)).cpu().numpy()
    temp_l1 = ((temp_l1-temp_l1.min())/(temp_l1.max()-temp_l1.min()) * 255.).astype('uint8')
    hist,bins = np.histogram(temp_l1.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    temp_l1_ = cdf[temp_l1]

    return temp_l1_
def make_maps_random(device):
    maps_list = []
    for i in range(2,9):
        if i ==2:
            maps_list.append(torch.rand([1,1,2**i,2**i], device = device))
        else:
            maps_list.append(torch.rand([1,1,2**i,2**i], device = device))
            maps_list.append(torch.rand([1,1,2**i,2**i], device = device))
    return maps_list

def make_maps_zero(device):
    maps_list = []
    for i in range(2,9):
        if i ==2:
            maps_list.append(torch.zero([1,1,2**i,2**i], device = device))
        else:
            maps_list.append(torch.zero([1,1,2**i,2**i], device = device))
            maps_list.append(torch.zero([1,1,2**i,2**i], device = device))
    return maps_list

def make_strs(G, device):
    strs_list = []
    for i in range(2,9):
        single_block = getattr(G.backbone.synthesis,f'b{2**i}')
        #import pdb;pdb.set_trace()
        strs_list.append(torch.Tensor(getattr(single_block,'conv1').noise_strength.detach().cpu().numpy()).to(device))
        #import pdb;pdb.set_trace()

        try:# getattr(single_block,'conv0'):
            strs_list.append(torch.Tensor(getattr(single_block,'conv0').noise_strength.detach().cpu().numpy()).to(device))
        except:
            pass
    return strs_list
#----------------------------------------------------------------------------
#XXX python gen_samples_style_mixing.py --outdir l1_map_diff --seeds 11-14
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default = '/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl')
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, metavar='BOOL', default=True, show_default=True)
@click.option('--part', help='coarse/middle/fine', type=click.Choice(['c','m','f']), metavar='STR', default='c', show_default=True)
@click.option('--mixing_type', help='w/map/map_indv', type=click.Choice(['w','map','map_indv']), metavar='STR', default='w', show_default=True)
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
    part: str,
    mixing_type: str,
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
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    ws_axis = None
    imgs_1st_row = []
    imgs_2nd_row = []
    if mixing_type == 'w':
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            
            PERTURB = .4

            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + 0, np.pi/2 + -.2, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2+PERTURB, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            
            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            if ws_axis is None:
                ws_axis = ws.clone()
            else:
                ws_axis_temp = ws_axis.clone()
                #XXX Which is coarse
                if part == 'c':
                    ws_axis_temp[:,:4] = ws.clone()[:,:4]
                #XXX Which is middle
                elif part == 'm':
                    ws_axis_temp[:,4:8] = ws.clone()[:,4:8]
                #XXX Which is fine
                elif part == 'f':
                    ws_axis_temp[:,8:] = ws.clone()[:,8:]
                
                img_swapped = G.synthesis(ws_axis_temp, None, camera_params)['image']
                img_swapped = (img_swapped.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs_2nd_row.append(img_swapped)
            #ws_prev = ws.clone()
            #import pdb;pdb.set_trace()
            img = G.synthesis(ws, None, camera_params)['image']
            
                

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            if seed_idx == 0:
                imgs_2nd_row.append(img)
                imgs_1st_row.append(img)
            else:
                imgs_1st_row.append(img)

        img_1st = torch.cat(imgs_1st_row, dim=2)
        img_2nd = torch.cat(imgs_2nd_row, dim=2)
        #import pdb;pdb.set_trace()
        img = torch.cat([img_1st, img_2nd], axis=1)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_{part}.png')

    elif mixing_type == 'map':

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            
            PERTURB = .4

            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + 0, np.pi/2 + -.2, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2+PERTURB, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            
            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

            img_bunch=[]
            l1_bunch = []
            for i in range(len(seeds)):
                temp_img = G.synthesis(ws,None, camera_params)['image']
                if len(img_bunch) !=0:
                    temp_l1 = torch.abs(temp_img.clone() - img_bunch[i-1])
                    #temp_l1 = torch.sum(temp_l1,dim=(0,1)).cpu().numpy()
                    #temp_l1 = ((temp_l1-temp_l1.min())/(temp_l1.max()-temp_l1.min()) * 255.).astype('uint8')
                    #hist,bins = np.histogram(temp_l1.flatten(),256,[0,256])
                    #cdf = hist.cumsum()
                    #cdf_normalized = cdf * float(hist.max()) / cdf.max()

                    #cdf_m = np.ma.masked_equal(cdf,0)
                    #cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
                    #cdf = np.ma.filled(cdf_m,0).astype('uint8')
                    #temp_l1_ = cdf[temp_l1]
                    temp_l1_ = histogram_equalizer(temp_l1)
                    #import pdb;pdb.set_trace()
                    l1_bunch.append(torch.stack([(torch.from_numpy(temp_l1_)/255.).to(device), (torch.from_numpy(temp_l1_)/255.).to(device), (torch.from_numpy(temp_l1_)/255.).to(device)], dim=0)[None, ...])
                else:
                    l1_bunch.append(torch.zeros_like(temp_img))
                img_bunch.append(temp_img)
            #import pdb;pdb.set_trace()
            imgs_random = torch.cat(img_bunch, dim=3)
            imgs_l1_map = torch.cat(l1_bunch, dim=3)
            #import pdb;pdb.set_trace()
            img = torch.cat([imgs_random , imgs_l1_map ], axis=2)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
    elif mixing_type == 'map_indv':

        maps_list = make_maps_random(device)
        

        strs_list = make_strs(G,device)
            
        
        #import pdb;pdb.set_trace()
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

            
            PERTURB = .4

            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + 0, np.pi/2 + -.2, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2+PERTURB, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            
            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

            
            #for i in range(len(seeds)):
            
            second_maps_list = make_maps_random(device)
            
            noise_axis= [10.*map_ for str_, map_ in zip(strs_list,maps_list)]
            import pdb;pdb.set_trace()
            img_axis = G.synthesis(ws, noise_axis, camera_params)['image']
            for j in range(13):
                save_list =[]
                map_store = maps_list[j].clone()
                maps_list[j] = second_maps_list[j].clone()
                #
                #noise_list_temp = [str_ * map_ for str_, map_ in zip(strs_list,maps_list)]
                noise_list_temp =[]
                for str_,map_ in zip(strs_list,maps_list):
                    noise_list_temp.append(str_*map_)
                #import pdb;pdb.set_trace()
                temp_img = G.synthesis(ws, noise_list_temp, camera_params)['image']
                maps_list[j] = map_store

                save_list.append(img_axis)
                save_list.append(temp_img)
                #save_list.append()
 
                temp_l1 = torch.sqrt(torch.abs(temp_img.clone() - img_axis.clone()))
                #import pdb;pdb.set_trace()
                temp_l1 = torch.sum(temp_l1, dim=1, keepdim=True).repeat(1,3,1,1)
                
                l1_map = (temp_l1-temp_l1.min())/(temp_l1.max()-temp_l1.min())
                #XXX histo
                #temp_l1_ = histogram_equalizer(temp_l1)
                #l1_map = torch.stack([(torch.from_numpy(temp_l1_)/255.).to(device), (torch.from_numpy(temp_l1_)/255.).to(device), (torch.from_numpy(temp_l1_)/255.).to(device)], dim=0)[None, ...]

                save_list.append(l1_map)

            #import pdb;pdb.set_trace()
            #imgs_random = torch.cat(img_bunch, dim=3)
            #imgs_l1_map = torch.cat(l1_bunch, dim=3)
            #import pdb;pdb.set_trace()
                img = torch.cat(save_list, dim=3)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_map{j}.png')



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
