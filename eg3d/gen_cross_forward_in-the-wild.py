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
@click.option('--network', 'network_pkl', help='Network pickle filename', default='/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl')
#@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR', default = './cross_forward')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, metavar='BOOL', default=True, show_default=True)
@click.option('--data_path', help='data path for paired smthng', type=str, default='/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d')

#XXX data_path: /home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined/
#XXX python gen_inversion.py --outdir=out_inversion_l1 --trunc=0.7 --seeds=0-3 --network=/home/nas4_user/jaeseonglee/CVPR2023/ckpts/pretrained_eg3d_ffhq256_025000.pkl --data_path /home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined
#XXX python gen_inversion.py --outdir=out_inversion_w_chanpretrain_inthewild --trunc=0.7 --seeds=0-3 --network=/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/ffhq512-128.pkl --data_path /home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/inthewild_data
#XXX CUDA_VISIBLE_DEVICES=2 python gen_inversion_in-the-wild.py --w_type w++ --network /home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl --seeds 0-3 --data_path inthewild_data --outdir ./out_inversion_w++&map_fixed_triplane 
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

    os.makedirs(outdir, exist_ok=True)



    path_imgs = []

    for i in os.listdir(os.path.join(data_path,'inthewild_data')):
        
        if i[-3:] == 'png':
            path_imgs.append(i)
        
    #import pdb;pdb.set_trace()
    predefined_names = ['candal.png']
    for idx, datum in enumerate(path_imgs):
        #np.random.seed(seed)
        if datum not in predefined_names:
            continue
        print(f'Cross forward for {datum}')
        #z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        with open(os.path.join(data_path, 'inthewild_data' ,'dataset.json')) as f:
            label = json.load(f)['labels']

        
        
        with open(os.path.join(data_path, 'wild_w_picked_pickles', datum+'.pickle'), 'rb') as f:
            w_pickle = pickle.load(f)
        
        with open(os.path.join(data_path, 'wild_w+_picked_pickles', datum+'.pickle'), 'rb') as f:
            wp_pickle = pickle.load(f)
            
        label = dict(label)
        #cam = label[datum]
        #c = torch.from_numpy(np.array(cam)).to(torch.float32).to(device).unsqueeze(0)
        #import pdb;pdb.set_trace()
        c = label[datum]
        c = torch.from_numpy(np.array(c)).to(torch.float32).to(device).unsqueeze(0)
        #c2 = label[datum2[:-7]]
        #c2 = torch.from_numpy(np.array(c2)).to(torch.float32).to(device).unsqueeze(0)


        
        w = w_pickle['ws'].to(device)
        wp = wp_pickle['ws'].to(device)
        #import pdb;pdb.set_trace()
        
        
  
        img_path  = os.path.join(data_path,'inthewild_data', datum)
        img = PIL.Image.open(img_path).resize((512,512), PIL.Image.LANCZOS)
        img = torch.from_numpy(np.array(img).transpose(2,0,1))
        img = (img.to(device).to(torch.float32) / 127.5 - 1).unsqueeze(0)

        
        

        # XXX XXX XXX XXX XXX XXX XXX XXX XXX #
        # XXX # w+ == #n of blocks * 2 XXX XXX #
        # XXX # w++ == #n of blocks * 3 - 1 XXX #
        # XXX XXX XXX XXX XXX XXX XXX XXX XXX #

     


        intrinsics = FOV_to_intrinsics(18.837, device=device)
    
        #for step in range(steps):
        if True:
    
            imgs_save = []
            result = G.synthesis_cross(wp, w.repeat(1,14,1), c=c)
                    
            img_rec = result['image']
            
            img_save = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_rec_save = (img_rec.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            
            imgs_save.append(img_save); imgs_save.append(img_rec_save)
            
            angle_p = -0.2
                
            for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                #ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                #import pdb;pdb.set_trace()
              
                result = G.synthesis_cross(wp, w.repeat(1,14,1), None, camera_params)
                img_temp = result['image']
                depth_temp = result['image_depth']
                img_temp_save = (img_temp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                depth_temp_save = (depth_temp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                imgs_save.append(img_temp_save)
            #img_save = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            #img_rec_save = (img_rec.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            #import pdb;pdb.set_trace()
            
            
            imgs_save_file = torch.cat(imgs_save, dim=2)
            #result = torch.cat((img_save, img_rec_save), dim=2)
            #import pdb;pdb.set_trace()
            PIL.Image.fromarray(imgs_save_file[0].cpu().numpy(), 'RGB').save(f'{outdir}/{datum}_cross_forward.png')
            #PIL.Image.fromarray(depth_temp_save[:,:,0].cpu().numpy(), 'L').save(f'{outdir}/{datum}_depth.png')
            
                #PIL.Image.fromarray(result[0].detach().cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_{step}.png')
            
        

        
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
