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
import pickle 

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import random
from torch_utils.ops import upfirdn2d
import json
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from training.vggloss import VGGLoss
from training.idloss import IDLoss
from training.dual_discriminator import filtered_resizing

#XXX python gen_samples.py --outdir=out --trunc=0.7 --shapes=true --seeds=0-3 --network=/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/ffhq512-128.pkl


def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#XXX python gen_inversion.py --outdir=out_inversion_l1_vgg_maps --trunc=0.7 --seeds=0-3 --network=/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/ffhq512-128.pkl --data_path /home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined --reload_modules True --loss_to full

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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True, default='/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl')

@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=False, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, metavar='BOOL', default=True, show_default=True)
@click.option('--data_path', help='data path for paired smthng', type=str, required=True, default='/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/out_inversion_w_chanpretrain_inthewild_fixed_triplane_pickles')
@click.option('--swap_type', help='Swap type map or w', type=click.Choice(['map', 'w', 'w_bunch']), default='map')
#XXX data_path: /home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined/
#XXX python gen_inversion.py --outdir=out_inversion_l1 --trunc=0.7 --seeds=0-3 --network=/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/ffhq512-128.pkl --data_path /home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined

def generate_images(
    network_pkl: str,

    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    data_path: str,
    swap_type: str,

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
        #import pdb;pdb.set_trace()
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new



    #intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    
    with open(os.path.join('/home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined','dataset_integrated_spherical.json')) as f:
        labels = json.load(f)['labels']
    labels = dict(labels)
    
    os.makedirs(os.path.join('/'.join(data_path.split('/')[:-1]), 'myswapped_ws+_bunch'),exist_ok=True)
    data = os.listdir(data_path)
    for datum in data:




        with open(os.path.join(data_path,datum), 'rb') as f:
            data1 = pickle.load(f)
        datum2 = random.choice(data)
        with open(os.path.join(data_path,datum2), 'rb') as g:
            data2 = pickle.load(g)

        pivot_key = 'mirror.png'
        while pivot_key.split('_')[-1] == 'mirror.png':
            pivot_key = np.random.choice(list(labels.keys()))
        
        c1 = labels[datum[:-7]]
        c1 = torch.from_numpy(np.array(c1)).to(torch.float32).to(device).unsqueeze(0)
        c2 = labels[datum2[:-7]]
        c2 = torch.from_numpy(np.array(c2)).to(torch.float32).to(device).unsqueeze(0)


        if swap_type == 'map':
            maps1 = [map.to(device) for map in data1['maps']]
            maps2 = [map.to(device) for map in data2['maps']]
            ws1 = data1['ws'].to(device)
            ws2 = data2['ws'].to(device)
            print(f'{datum}&{datum2} started!')
            for i in range(len(maps1)):
                #import pdb;pdb.set_trace()
                img_rec1 = G.synthesis(ws1, maps1, c1)['image']
                img_rec2 = G.synthesis(ws2, maps2, c2)['image']
                img_rec1_save = (img_rec1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_rec2_save = (img_rec2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                temp1 = maps1.clone()[i]
                temp2 = maps2.clone()[i]
                maps1[i] = temp2
                maps2[i] = temp1
                img_swap1 = G.synthesis(ws1, maps1, c1)['image']
                img_swap2 = G.synthesis(ws2, maps2, c2)['image']
                img_swap1_save = (img_swap1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_swap2_save = (img_swap2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                
                img_l1_map1 = (img_rec1 - img_swap1)
                img_l1_map1 = (img_l1_map1 - img_l1_map1.min())/(img_l1_map1.max() - img_l1_map1.min())

                img_l1_map2 = (img_rec2 - img_swap2)
                img_l1_map2 = (img_l1_map2 - img_l1_map2.min())/(img_l1_map2.max() - img_l1_map2.min())

                result = torch.cat((img_rec1_save, img_rec2_save, img_swap1_save, img_swap2_save, img_l1_map1, img_l1_map2), dim=2)
                maps1[i] = temp1
                maps2[i] = temp2
                
                PIL.Image.fromarray(result[0].detach().cpu().numpy(), 'RGB').save(os.path.join('/'.join(data_path.split('/')[:-1]), 'myswapped_ws+_bunch+')+f'/{datum}_and_{datum2}_{i}_swapped.png')
       
        elif swap_type == 'w':
            #maps1 = [map.to(device) for map in data1['maps']]
            #maps2 = [map.to(device) for map in data2['maps']]
            ws1 = data1['ws'].to(device)
            ws2 = data2['ws'].to(device)
            print(f'{datum.split(".")[0]}&{datum2.split(".")[0]} started!')
            assert(ws1.shape[1] == 14)
            for i in range(ws1.shape[1]):
                #import pdb;pdb.set_trace()
                img_rec1 = G.synthesis(ws1, None, c1)['image']
                img_rec2 = G.synthesis(ws2, None, c2)['image']
                img_rec1_save = (img_rec1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_rec2_save = (img_rec2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                #temp1 = ws1[0][i]
                #temp2 = ws2[0][i]
                temp_ws1 = ws1.clone()
                temp_ws2 = ws2.clone()
                #import pdb;pdb.set_trace()
                temp_ws1[0][i] = ws2[0][i]
                temp_ws2[0][i] = ws1[0][i]
                img_swap1 = G.synthesis(temp_ws1, None, c1)['image']
                img_swap2 = G.synthesis(temp_ws2, None, c2)['image']
                img_swap1_save = (img_swap1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_swap2_save = (img_swap2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                result = torch.cat((img_rec1_save, img_rec2_save, img_swap1_save, img_swap2_save), dim=2)
                #ws1[0][i] = temp1
                #ws2[0][i] = temp2
                
                PIL.Image.fromarray(result[0].detach().cpu().numpy(), 'RGB').save(os.path.join('/'.join(data_path.split('/')[:-1]), 'myswapped_ws+_bunch+')+f'/{datum.split(".")[0]}_and_{datum2.split(".")[0]}_{i}_swapped.png')
    
        elif swap_type == 'w_bunch':
            #maps1 = [map.to(device) for map in data1['maps']]
            #maps2 = [map.to(device) for map in data2['maps']]
            ws1 = data1['ws'].to(device)
            ws2 = data2['ws'].to(device)
            print(f'{datum.split(".")[0]}&{datum2.split(".")[0]} started!')
            assert(ws1.shape[1] == 14)
            for i in range(ws1.shape[1]//2):
                #import pdb;pdb.set_trace()
                img_rec1 = G.synthesis(ws1, None, c1)['image']
                img_rec2 = G.synthesis(ws2, None, c2)['image']
                img_rec1_save = (img_rec1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_rec2_save = (img_rec2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                #temp1 = ws1[0][i]
                #temp2 = ws2[0][i]
                temp_ws1 = ws1.clone()
                temp_ws2 = ws2.clone()
                #import pdb;pdb.set_trace()
                temp_ws1[0][i*2] = ws2[0][i*2]
                temp_ws1[0][i*2+1] = ws2[0][i*2+1]
                temp_ws2[0][i*2] = ws1[0][i*2]
                temp_ws2[0][i*2+1] = ws1[0][i*2+1]
                img_swap1 = G.synthesis(temp_ws1, None, c1)['image']
                img_swap2 = G.synthesis(temp_ws2, None, c2)['image']
                img_swap1_save = (img_swap1.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_swap2_save = (img_swap2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                result = torch.cat((img_rec1_save, img_rec2_save, img_swap1_save, img_swap2_save), dim=2)
                #ws1[0][i] = temp1
                #ws2[0][i] = temp2
                
                PIL.Image.fromarray(result[0].detach().cpu().numpy(), 'RGB').save(os.path.join('/'.join(data_path.split('/')[:-1]), 'myswapped_ws+_bunch+')+f'/{datum.split(".")[0]}_and_{datum2.split(".")[0]}_{i}_swapped.png')
    


            

  
        
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
