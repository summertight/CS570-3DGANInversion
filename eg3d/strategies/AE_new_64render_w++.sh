echo 'AE_new train start'

CUDA_VISIBLE_DEVICES=0,2 python train.py --outdir=./training-runs-ae_new_64_1129 \
--cfg=ffhq --data=/home/nas2_userG/junhahyung/FFHQ_png_512.zip \
--gpus=2 --batch 16 --gamma 1 --gen_pose_cond True \
--loss_selection l1 1 --loss_selection id .1 --loss_selection vgg .8 \
--workers 4 --pretrain /home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl \
--w_type w++ --mode AE_new --mbstd-group 4 \
--lr_g 0.0001 --lr_d 0.00008 --lr_e 0.0001 \
--neural_rendering_resolution_initial 64