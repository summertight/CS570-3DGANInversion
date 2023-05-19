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
from training.aligner import align_and_crop_with_5points_tensor


from training.arcface_torch.backbones import get_model
from training.arcface_torch.utils.utils_config import get_config


#from facenet_pytorch import MTCNN
from training.facenet_pytorch_folder.models.mtcnn import MTCNN
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
def denorm(x):
    return (x / 2 + 0.5).clamp(0, 1)

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
import sys
sys.path.append('./training/DECA')

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl')
#@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR', default = './out_candal_multi_shape')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, metavar='BOOL', default=True, show_default=True)
@click.option('--data_path', help='data path for paired smthng', type=str, default='/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/inthewild_data')
#@click.option('--loss_choice', help='Select Losses', type=click.Choice(['l1', 'vgg', 'id','feat_32']), default=['l1','vgg'],multiple=True)
@click.option('--loss_choice', type=(str,float) , default =[('l1',1),('deca',5)],multiple=True)
@click.option('--lr', help='lr', type=float, default=3e-3)
@click.option('--steps', help='steps', type=int, default=50001)
@click.option('--src_name', type=str, required=True)
@click.option('--tgt_name', type=str, required=True)
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

    reload_modules: bool,
    data_path: str,
    loss_choice: str,
  
    lr: float,
    steps: int,
    src_name: str,
    tgt_name: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    loss_choice = dict(loss_choice)

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

    #intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    print(loss_choice)
    if 'vgg' in loss_choice:
        criterion_VGG = VGGLoss().to(device)
    if 'id' in loss_choice:
        criterion_ID = IDLoss().to(device)
    if 'deca_shape' in loss_choice or 'deca_exp' in loss_choice:
        from training.DECA.decalib.deca import DECA
        from training.DECA.decalib.utils.tensor_cropper import Cropper
        from training.DECA.decalib.datasets.detectors import batch_FAN
        from training.DECA.decalib.utils.config import cfg as deca_cfg
        from training.mobile_face_net import load_face_landmark_detector
        criterion_DECA = DECA(config=deca_cfg, device=device)
        landmark_detector = load_face_landmark_detector()
        landmark_detector = landmark_detector.to(device)
        landmark_detector.eval()
   
    config  = '/home/nas1_userC/jooyeolyun/repos/insightface/recognition/arcface_torch/configs/wf42m_pfc_r50.py'
    cfg = get_config(config)
    backbone = get_model(cfg.network, dropout=0.0, fp16=False, num_features=cfg.embedding_size)
    pp='/home/nas1_userC/jooyeolyun/repos/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc_r50/model.pt'
    backbone.load_state_dict(torch.load(pp, map_location='cpu'))
    backbone.eval().cuda()


    #    labels = json.load(f)['labels']
    resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
 

    path_imgs = []

    for i in os.listdir(data_path):
        
        if i[-3:] == 'png':
            path_imgs.append(i)
        
    

    #src_name = 'candal.png'
    #tgt_name = 'dua.png'
    data_path = '/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/inthewild_data'
    data_rand_path =data_path.replace('inthewild_data','sample_refined')
    ldmk_src = np.load(os.path.join(data_path.replace('_data','_data_kps'),src_name+'.npy'),allow_pickle=True)
    ldmk_src = ldmk_src.astype(np.float32)
    ldmk_tgt = np.load('/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/sample_refined_kps/seed0001_0.png.npy',allow_pickle=True)
    ldmk_tgt = ldmk_tgt.astype(np.float32)
    #breakpoint()
    if True:
       
        print(f'Face Swapping for source {src_name} & target {tgt_name}')
        
        with open(os.path.join(data_path, 'dataset.json')) as f:
            label = json.load(f)['labels']
        
        label = dict(label)
        cam_src = label[src_name]
        cam_tgt = np.load('/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/sample_refined/seed0001_0.npy')
        #breakpoint()
        c_tgt = torch.from_numpy(np.array(cam_tgt)).to(torch.float32).to(device)
  
        img_src_path  = os.path.join(data_path, src_name)
        img_tgt_path  = os.path.join(data_path, tgt_name)

        img_src = PIL.Image.open(img_src_path).resize((512,512), PIL.Image.LANCZOS)
        img_src = torch.from_numpy(np.array(img_src).transpose(2,0,1))
        img_src = (img_src.to(device).to(torch.float32) / 127.5 - 1).unsqueeze(0)

        img_tgt = PIL.Image.open('/home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/sample_refined/seed0001_0.png').resize((512,512), PIL.Image.LANCZOS)
        img_tgt = torch.from_numpy(np.array(img_tgt).transpose(2,0,1))
        img_tgt = (img_tgt.to(device).to(torch.float32) / 127.5 - 1).unsqueeze(0)
        #parse_path = '/home/nas4_user/jaeseonglee/ICCV2023/PSFR-GAN/inthewild_parse'
        #tgt_mask = PIL.Image.open(os.path.join(parse_path,tgt_name[:-4],'hq_final_binaryback.jpg'))
    
        #msk_tgt = torch.from_numpy(np.array(tgt_mask)[...,None].transpose(2,0,1))
        #msk_tgt = (msk_tgt.to(device).to(torch.float32) / 255).unsqueeze(0)
        
        #src_mask = PIL.Image.open(os.path.join(parse_path,src_name[:-4],'hq_final_binaryback.jpg'))
    
        #msk_src = torch.from_numpy(np.array(src_mask)[...,None].transpose(2,0,1))
        #msk_src = (msk_src.to(device).to(torch.float32) / 255).unsqueeze(0)
        
        tgt_mask_path='/home/nas4_user/jaeseonglee/ICCV2023/PSFR-GAN/w_unalign_results/seed0001_0/hq_final_binarylq.jpg'
        tgt_mask = PIL.Image.open(tgt_mask_path)
        msk_tgt = torch.from_numpy(np.array(tgt_mask)[...,None].transpose(2,0,1))
        msk_tgt = (msk_tgt.to(device).to(torch.float32) / 255).unsqueeze(0)
        
        loss = 0

        

        angle_p = -0.2
        
        
        
        w = G.backbone.mapping.w_avg.unsqueeze(0).unsqueeze(1).repeat(1,14,1).detach().cpu().cuda()
     

        w.requires_grad = True

        # XXX XXX XXX XXX XXX XXX XXX XXX XXX #
        # XXX # w+ == #n of blocks * 2 XXX XXX #
        # XXX # w++ == #n of blocks * 3 - 1 XXX #
        # XXX XXX XXX XXX XXX XXX XXX XXX XXX #

       
        optm = torch.optim.Adam(
            [w],#+maps,
            lr=lr)



        global_name = f"{src_name}_{'seed0001_0'}_{str(loss_choice)}"

        os.makedirs(os.path.join(outdir, global_name), exist_ok=True)

        intrinsics = FOV_to_intrinsics(18.837, device=device)
        mtcnn = MTCNN(image_size=(256), margin=0,device=device)
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
        camera_params_canonical = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        
        #mtcnn.to(device)
        for step in range(steps):
            loss = 0
            
            result, feat_list = G.synthesis_extract_feat(w, c=c_tgt)
         
            

            img_rec = result['image']
            
            
            img_rec_raw = result['image_raw']
            img_raw = filtered_resizing(img_tgt, size=128, f=resample_filter, filter_mode='antialiased')
        
    
            img_depth_rec = result['image_depth']
            loss_vgg=None; loss_id=None; loss_l1=None; loss_vgg_raw=None; loss_id_raw=None; loss_l1_raw=None
            
            img_rec_mini = torch.nn.functional.interpolate(img_rec,size=(256,256))
            img_tgt_mini = torch.nn.functional.interpolate(img_tgt,size=(256,256))
            img_src_mini = torch.nn.functional.interpolate(img_src,size=(256,256))
            #breakpoint()
            _, ldmk_src_tensor = mtcnn.detect_tensor((img_src_mini+1)/2*255)
            _, ldmk_tgt_tensor = mtcnn.detect_tensor((img_tgt_mini+1)/2*255)
            _, ldmk_rec_tensor = mtcnn.detect_tensor((img_rec_mini+1)/2*255)
            #img_src_aligned = align_and_crop_with_5points_tensor(img_src_mini, torch.tensor(ldmk_src,device=device).squeeze(1))
            #img_rec_aligned = align_and_crop_with_5points_tensor(img_rec_mini, torch.tensor(ldmk_tgt,device=device).squeeze(1))
            #img_tgt_aligned = align_and_crop_with_5points_tensor(img_tgt_mini, torch.tensor(ldmk_tgt,device=device).squeeze(1))
            img_src_aligned = align_and_crop_with_5points_tensor(img_src_mini, ldmk_src_tensor)
            img_rec_aligned = align_and_crop_with_5points_tensor(img_rec_mini, ldmk_rec_tensor)
            img_tgt_aligned = align_and_crop_with_5points_tensor(img_tgt_mini, ldmk_tgt_tensor)
            
            img_rec_aligned_224 = torch.nn.functional.interpolate(img_rec_aligned ,size=(112,112))
            img_tgt_aligned_224 = torch.nn.functional.interpolate(img_tgt_aligned ,size=(112,112))
            img_src_aligned_224 = torch.nn.functional.interpolate(img_src_aligned ,size=(112,112))
            
            loss_logger = {}

            if 'l2_mask' in loss_choice:
                msk_tgt[msk_tgt == 0] = .1
                loss_l2_mask = torch.nn.functional.mse_loss(img_rec*msk_tgt, img_tgt*msk_tgt).mean()
                loss+= loss_choice['l2_mask'] * loss_l2_mask

                loss_logger['l2_mask'] = loss_l2_mask
            if 'l1' in loss_choice:
                loss_l1 = torch.nn.functional.l1_loss(img_rec, img_tgt).mean()
                loss += loss_choice['l1']*loss_l1

                loss_logger['l1'] = loss_l1

            if 'l1_raw' in loss_choice:
                loss_l1_raw = torch.nn.functional.l1_loss(img_rec_raw, img_raw).mean()
                loss += loss_choice['l1_raw']* loss_l1_raw

                loss_logger['l1_raw'] = loss_l1_raw


            if 'vgg' in loss_choice:
               
                
                loss_vgg = criterion_VGG(img_rec_mini, img_tgt_mini).mean()
                loss += loss_choice['vgg']* loss_vgg
                
                loss_logger['vgg'] = loss_vgg
                    
            
            if 'id' in loss_choice:
                
               
                img_src_aligned = align_and_crop_with_5points_tensor(img_src_mini, torch.tensor(ldmk_src,device=device).squeeze(1))
                img_rec_aligned = align_and_crop_with_5points_tensor(img_rec_mini, torch.tensor(ldmk_tgt,device=device).squeeze(1))
                img_tgt_aligned = align_and_crop_with_5points_tensor(img_tgt_mini, torch.tensor(ldmk_tgt,device=device).squeeze(1))
                
                loss_id = criterion_ID(img_src_aligned , img_rec_aligned).mean()
                loss += loss_choice['id']* loss_id

                loss_logger['id'] = loss_id

            if 'id_yjl' in loss_choice:
                #breakpoint()
                id_feat_src = backbone(img_src_aligned_224)
                id_feat_rec = backbone(img_rec_aligned_224)
                loss_id_yjl = (1-torch.cosine_similarity(id_feat_src, id_feat_rec)).mean()
                loss += loss_choice['id_yjl']* loss_id_yjl

                loss_logger['id_yjl'] = loss_id_yjl

            if 'id_yjl_canonical' in loss_choice:
                #breakpoint()

                
                result_canonical = G.synthesis(w, None, c=camera_params_canonical)
                img_rec_canonical = result_canonical['image']
                img_rec_canonical_mini = torch.nn.functional.interpolate(img_rec_canonical, size=(256,256))

                _, ldmk_rec_canonical_tensor = mtcnn.detect_tensor((img_rec_canonical_mini+1)/2*255)
                img_rec_canonical_aligned = align_and_crop_with_5points_tensor(img_rec_canonical_mini, ldmk_rec_canonical_tensor)
                img_rec_canonical_aligned_224= torch.nn.functional.interpolate(img_rec_canonical_aligned ,size=(112,112))
                
                id_feat_src = backbone(img_src_aligned_224)
                id_feat_rec_canonical = backbone(img_rec_canonical_aligned_224)
                loss_id_yjl_canonical = (1-torch.cosine_similarity(id_feat_src, id_feat_rec_canonical)).mean()
                loss += loss_choice['id_yjl_canonical']* loss_id_yjl_canonical

                loss_logger['id_yjl_canonical'] = loss_id_yjl_canonical


                

            if step%100==0:
                #if 'vgg' in loss_choice:
                print('\n')
                print(f'src: {src_name} & tgt: {tgt_name}')
                for k in loss_logger.keys():
                    print(f"{k}: {loss_logger[k]}")
                print('\n')
                imgs_save = []
                img_src_save = (img_src.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_tgt_save = (img_tgt.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_rec_save = (img_rec.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
               
               
                imgs_save.append(img_src_save)
                imgs_save.append(img_tgt_save)
                imgs_save.append(img_rec_save)

                imgs_aligned_save = []
                img_rec_canonical_aligned_save = (img_rec_canonical_aligned.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_src_aligned_save = (img_src_aligned.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_tgt_aligned_save = (img_tgt_aligned.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                

                imgs_aligned_save.append(img_rec_canonical_aligned_save)
                imgs_aligned_save.append(img_src_aligned_save)
                imgs_aligned_save.append(img_tgt_aligned_save)
                
                angle_p = -0.2

        
                
                for idx, (angle_y, angle_p) in enumerate([(.4, angle_p), (0, angle_p), (-.4, angle_p)]):
                    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    #ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                    #import pdb;pdb.set_trace()
                    
                    result_temp = G.synthesis(w,None, camera_params)
                    img_temp = result_temp['image']
                    

                    img_temp_save = (img_temp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    imgs_save.append(img_temp_save)
                
                #breakpoint()
                imgs_save_file = torch.cat(imgs_save, dim=2)
                imgs_save_file_aligned = torch.cat(imgs_aligned_save, dim=2)
                #breakpoint()
                #depths_save_file = torch.cat(depths_save, dim=2).repeat(1,1,1,3)
                #result = torch.cat((img_save, img_rec_save), dim=2)
                #import pdb;pdb.set_trace()
                PIL.Image.fromarray(imgs_save_file[0].cpu().numpy(), 'RGB').save(f'{outdir}/{global_name}/{step}.png')
                PIL.Image.fromarray(imgs_save_file_aligned[0].cpu().numpy(), 'RGB').save(f'{outdir}/{global_name}/{step}_aligned.png')
                
                #PIL.Image.fromarray(depths_save_file[0].cpu().numpy(), 'RGB').save(f'{outdir}/{global_name}/{step}_depth.png')
                
                #PIL.Image.fromarray(result[0].detach().cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_{step}.png')
            
            if step == steps-1:
                print('\nSaving optimized ws!!\n')
                
                
                data = {
                'ws': w.detach().cpu()
                }
                # save
                os.makedirs(f'{outdir}_pickles',exist_ok=True)
                with open(f'{outdir}/{global_name}/{step}.pickle', 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

            optm.zero_grad()
            loss.backward()
            optm.step()
            #sche.step()

        
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
