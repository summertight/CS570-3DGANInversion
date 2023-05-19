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
def denorm(x):
    return (x / 2 + 0.5).clamp(0, 1)

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
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, metavar='BOOL', default=True, show_default=True)
@click.option('--loss_choice', type=(str,float) , default =[('l1',1),('deca',5)],multiple=True)
@click.option('--lr', help='lr', type=float, default=1e-3)
@click.option('--steps', help='steps', type=int, default=10001)
@click.option('--src_img_path', type=str, required=True)
@click.option('--tgt_vid_path', type=str, required=True)
@click.option('--frames', type=int, required=True)
@click.option('--outdir', type=str,required=True)
#XXX data_path: /home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined/
#XXX python gen_inversion.py --outdir=out_inversion_l1 --trunc=0.7 --seeds=0-3 --network=/home/nas4_user/jaeseonglee/CVPR2023/ckpts/pretrained_eg3d_ffhq256_025000.pkl --data_path /home/nas1_userB/dataset/ffhq-dataset/images1024x1024/images1024x1024_refined
#XXX python gen_inversion.py --outdir=out_inversion_w_chanpretrain_inthewild --trunc=0.7 --seeds=0-3 --network=/home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/ffhq512-128.pkl --data_path /home/nas4_user/jaeseonglee/ICCV2023/eg3d/eg3d/inthewild_data
#XXX CUDA_VISIBLE_DEVICES=2 python gen_inversion_in-the-wild.py --w_type w++ --network /home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl --seeds 0-3 --data_path inthewild_data --outdir ./out_inversion_w++&map_fixed_triplane 
def generate_images(
    network_pkl: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    reload_modules: bool,
    loss_choice: str,
    lr: float,
    steps: int,
    src_img_path: str,
    tgt_vid_path: str,
    frames: int,
    outdir: str
):

    # XXX XXX XXX XXX XXX XXX XXX XXX XXX #
    # XXX # w+ == #n of blocks * 2 XXX XXX #
    # XXX # w++ == #n of blocks * 3 - 1 XXX #
    # XXX XXX XXX XXX XXX XXX XXX XXX XXX #

    device = torch.device('cuda')
    loss_choice = dict(loss_choice)
    resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)

    print('Loading networks from "%s"...' % network_pkl)
    
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




    print(f'MY LOSS CHOICES ARE {loss_choice} !')

    if 'vgg' in loss_choice:
        criterion_VGG = VGGLoss().to(device)

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

    if 'id' in loss_choice or 'id_yjl' in loss_choice:
        config  = '/home/nas1_userC/jooyeolyun/repos/insightface/recognition/arcface_torch/configs/wf42m_pfc_r50.py'
        cfg = get_config(config)
        backbone = get_model(cfg.network, dropout=0.0, fp16=False, num_features=cfg.embedding_size)
        pp='/home/nas1_userC/jooyeolyun/repos/insightface/recognition/arcface_torch/work_dirs/wf42m_pfc_r50/model.pt'
        backbone.load_state_dict(torch.load(pp, map_location='cpu'))
        backbone.eval().to(device)

    mtcnn = MTCNN(image_size=(256), margin=0, device=device)
        #mtcnn.to(device)
        
    if 'deca_shape' in loss_choice or 'deca_exp' in loss_choice:
        import sys
        sys.path.append('./training/DECA')
        from training.DECA.decalib.deca import DECA
        from training.DECA.decalib.utils.tensor_cropper import Cropper
        from training.DECA.decalib.datasets.detectors import batch_FAN
        from training.DECA.decalib.utils.config import cfg as deca_cfg
        from training.mobile_face_net import load_face_landmark_detector
        criterion_DECA = DECA(config=deca_cfg, device=device)
        landmark_detector = load_face_landmark_detector()
        landmark_detector = landmark_detector.to(device)
        landmark_detector.eval()


        
    

    print(f'LOADING SOURCE IMG"s assets')

    ldmk_src = np.load((src_img_path+'.npy').replace('_data','_data_kps'),allow_pickle=True)
    ldmk_src = ldmk_src.astype(np.float32)

    img_src = PIL.Image.open(src_img_path).resize((512,512), PIL.Image.LANCZOS)
    img_src = torch.from_numpy(np.array(img_src).transpose(2,0,1))
    img_src = (img_src.to(device).to(torch.float32) / 127.5 - 1).unsqueeze(0)
    img_src_mini = torch.nn.functional.interpolate(img_src,size=(256,256))
    #breakpoint()
    _, ldmk_src_tensor = mtcnn.detect_tensor((img_src_mini+1)/2*255)
    img_src_aligned = align_and_crop_with_5points_tensor(img_src_mini, ldmk_src_tensor)
    img_src_aligned_112 = filtered_resizing(img_src_aligned, size=112, f=resample_filter, filter_mode='antialiased')

    print('DONE!')

    print(f'LOADING TARGET FRAMES" assets')

    frms_tgt_dict = {}
    c_tgt_dict = {}
    ldmks_tgt_dict={}
    masks_tgt_dict={}
    
    with open(os.path.join(tgt_vid_path, 'dataset.json')) as f:
            label = json.load(f)['labels']
    label = dict(label)

    data_counter=0
    for tgt_name in os.listdir(tgt_vid_path):

        if tgt_name[-3:] != 'png':
            continue
        if data_counter == frames:
            break
        #XXX LDMK
        ldmks_tgt_dict[tgt_name] = np.load(os.path.join(tgt_vid_path.replace('crop','crop_kps'),tgt_name+'.npy')).astype(np.float32)
        
        #XXX FRAME
        frm_tgt_temp = PIL.Image.open(os.path.join(tgt_vid_path,tgt_name)).resize((512,512), PIL.Image.LANCZOS)
        frm_tgt_temp = torch.from_numpy(np.array(frm_tgt_temp).transpose(2,0,1))
        frm_tgt_temp = (frm_tgt_temp.to(device).to(torch.float32) / 127.5 - 1).unsqueeze(0)
        frms_tgt_dict[tgt_name] = frm_tgt_temp

        #XXX CAM
        cam_tgt_temp = label[tgt_name]
        c_tgt_temp = torch.from_numpy(np.array(cam_tgt_temp)).to(torch.float32).to(device).unsqueeze(0)
        c_tgt_dict[tgt_name] = c_tgt_temp
        
        #XXX mask
        msk_tgt_temp = PIL.Image.open(os.path.join(tgt_vid_path,tgt_name).replace('crop','mask')).resize((512,512), PIL.Image.LANCZOS)
        msk_tgt_temp = torch.from_numpy(np.array(msk_tgt_temp)[...,None].transpose(2,0,1))
        msk_tgt_temp = (msk_tgt_temp.to(device).to(torch.float32) / 127.5 - 1).unsqueeze(0)
        masks_tgt_dict[tgt_name] = msk_tgt_temp


        data_counter+=1

    frms_tgt = [frm for frm in frms_tgt_dict.values()]
    frms_tgt = torch.cat(frms_tgt,0)
    frms_tgt_raw = filtered_resizing(frms_tgt, size=128, f=resample_filter, filter_mode='antialiased')
    frms_tgt_mini = filtered_resizing(frms_tgt, size=256, f=resample_filter, filter_mode='antialiased')

    _, ldmk_tgt_tensor = mtcnn.detect_tensor((frms_tgt_mini+1)/2*255)
    #breakpoint()
    frms_tgt_aligned = align_and_crop_with_5points_tensor(frms_tgt_mini, ldmk_tgt_tensor)
    frms_tgt_aligned_112 = filtered_resizing(frms_tgt_aligned, size=112, f=resample_filter, filter_mode='antialiased')
    #breakpoint()

    cams_tgt = [cam for cam in c_tgt_dict.values()]
    cams_tgt = torch.cat(cams_tgt,0)

    ldmks_tgt = [torch.tensor(ldmk,device=device) for ldmk in ldmks_tgt_dict.values()]
    ldmks_tgt = torch.cat(ldmks_tgt,0)

    msks_tgt = [msk for msk in masks_tgt_dict.values()]
    msks_tgt = torch.cat(msks_tgt,0)
   
    #breakpoint()
   
    print('DONE!')
    
    print('SETTING SOME UTILS')
    src_img_name = src_img_path.split('/')[-1]
    tgt_vid_name = tgt_vid_path.split('/')[-1]
    global_name = f"{src_img_name}_{tgt_vid_name}_{str(loss_choice)}"
    intrinsics = FOV_to_intrinsics(18.837, device=device)
    
    #w_delta 
    w = G.backbone.mapping.w_avg.unsqueeze(0).unsqueeze(1).repeat(1,14,1).detach().cpu().cuda()
    if frames!=1:
        w_delta = torch.nn.Parameter(torch.zeros_like(w.repeat(frames, 1, 1),device=device))
    w.requires_grad = True
    print("DONE")
    #breakpoint()

    if True:

        loss = 0
        if frames!=1:
            optm = torch.optim.Adam(
                [w,w_delta],#+maps,
                lr=lr)
        else:
            optm = torch.optim.Adam(
                [w],#+maps,
                lr=lr)


        
        for step in range(steps):
            loss = 0
            if frames!=1:
                result, feat_list = G.synthesis_extract_feat(w.repeat(cams_tgt.shape[0],1,1) + w_delta, c=cams_tgt)
            else:
                result, feat_list = G.synthesis_extract_feat(w.repeat(cams_tgt.shape[0],1,1), c=cams_tgt)
           
            #print("PREPARING RESULT INGREDIENTS for CALC LOSSES")

            frms_rec = result['image']
            frms_rec_raw = result['image_raw']
            frms_rec_mini = filtered_resizing(frms_rec, size=256, f=resample_filter, filter_mode='antialiased')
            
         
    
            
            
            #_, ldmk_rec_tensor = mtcnn.detect_tensor((frms_rec_mini+1)/2*255)

            #frms_rec_aligned = align_and_crop_with_5points_tensor(frms_rec_mini, ldmk_rec_tensor)
            frms_rec_aligned = align_and_crop_with_5points_tensor(frms_rec_mini, ldmks_tgt)
            frms_rec_aligned_112 = filtered_resizing(frms_rec_aligned, size=112, f=resample_filter, filter_mode='antialiased')
            
            
            
            loss_logger = {}

            if 'l1' in loss_choice:
                loss_l1 = torch.nn.functional.l1_loss(frms_rec, frms_tgt).mean()
                loss += loss_choice['l1']*loss_l1

                loss_logger['l1'] = loss_l1

     
           

            if 'vgg' in loss_choice:
               
                
                loss_vgg = criterion_VGG(frms_rec, frms_tgt).mean()
                loss += loss_choice['vgg']* loss_vgg
                
                loss_logger['vgg'] = loss_vgg
                    
            


            if 'id_yjl' in loss_choice:
                #breakpoint()
                id_feat_src = backbone(img_src_aligned_112.repeat(frames, 1, 1, 1))
                id_feat_rec = backbone(frms_rec_aligned_112)
                loss_id_yjl = (1-torch.cosine_similarity(id_feat_src, id_feat_rec)).mean()
                loss += loss_choice['id_yjl']* loss_id_yjl

                loss_logger['id_yjl'] = loss_id_yjl
            
            if 'w_reg' in loss_choice:

                loss_w_reg = torch.norm(w_delta,dim=1).mean()
                loss+=loss_choice['w_reg']*loss_w_reg
                loss_logger['w_reg'] =loss_w_reg

            if 'l2_mask' in loss_choice:
                
                
                msks_tgt[msks_tgt == 0] = .2
                loss_l2_mask = torch.nn.functional.mse_loss(frms_rec*msks_tgt, frms_tgt*msks_tgt).mean()
                loss+= loss_choice['l2_mask'] * loss_l2_mask

                loss_logger['l2_mask'] = loss_l2_mask
           # breakpoint()
            if 'deca_exp' in loss_choice:
                frms_tgt_cropped = landmark_detector.align_face(
                    inputs=denorm(frms_tgt), scale=1.25, inverse=False, target_size=224)
            

                codedict_tgt= criterion_DECA.encode(frms_tgt_cropped,use_detail=False)#XXX (0,1)
                
                frms_rec_cropped = landmark_detector.align_face(
                    inputs=denorm(frms_rec), scale=1.25, inverse=False, target_size=224)

                codedict_rec = criterion_DECA.encode(frms_rec_cropped,use_detail=False)
                breakpoint()
                loss_3dmm_exp = torch.nn.functional.l1_loss(codedict_tgt['exp'], codedict_rec['exp'])
                loss += loss_choice['deca_exp']*loss_3dmm_exp

                loss_logger['deca_exp'] = loss_3dmm_exp
        
          
                

            if step%100==0:
                #if 'vgg' in loss_choice:
                print('\n')
                print('#'*30)
                print(f'{str(step)}/{str(steps)}')
                print(f'src: {src_img_name} & tgt: {tgt_vid_name}')
                for k in loss_logger.keys():
                    print(f"{k}: {loss_logger[k]}")
                print('#'*30)
                print('\n')
                first_row_save = []
                second_row_save = []
                img_src_save = (img_src.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                frms_tgt_save = (frms_tgt.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                frms_rec_save = (frms_rec.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                frms_rec_aligned_save = (frms_rec_aligned_112.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                first_row_save.append(img_src_save)
                second_row_save.append(torch.zeros_like(img_src_save))
                for i in range(frames):
                    first_row_save.append(frms_tgt_save[i:i+1])
                    second_row_save.append(frms_rec_save[i:i+1])

                align_check = []
                #breakpoint()
                for i in range(frames):
                    align_check.append(frms_rec_aligned_save[i:i+1])
                
                
                #breakpoint()
     
                angle_p = -0.2

             
                imgs_multiview_save=[]

                for idx, (angle_y, angle_p) in enumerate([(.4, angle_p), (0, angle_p), (-.4, angle_p)]):
                    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                    #ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                    #import pdb;pdb.set_trace()
                    
                    result_temp = G.synthesis(w , None , camera_params)
                    img_temp = result_temp['image']
                    

                    img_temp_save = (img_temp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    imgs_multiview_save.append(img_temp_save)
                
                #breakpoint()
                imgs_save_file_mv = torch.cat(imgs_multiview_save, dim=2)
                imgs_save_file = torch.cat((torch.cat(first_row_save,2),torch.cat(second_row_save,2)),1)
                align_check_save = torch.cat(align_check, dim=2)
                #breakpoint()
                os.makedirs(f'{outdir}/{global_name}',exist_ok=True)
                PIL.Image.fromarray(imgs_save_file[0].cpu().numpy(), 'RGB').save(f'{outdir}/{global_name}/{step}.png')
                PIL.Image.fromarray(imgs_save_file_mv[0].cpu().numpy(), 'RGB').save(f'{outdir}/{global_name}/{step}_mv.png')
                PIL.Image.fromarray(align_check_save.detach().cpu().numpy()[0], 'RGB').save(f'{outdir}/{global_name}/{step}_align_check.png')
               
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
