echo 'AE_new train start'

CUDA_VISIBLE_DEVICES=0,1,2 \
python train.py --outdir=/home/nas2_userF/gyojunggu/gyojung/faceswap/eg3d/eg3d/ae_new_128_w+_1202_noid_yesgan_1:2_noTV \
--gpus=3 --batch 6 --gamma 1 \
--loss_selection l1 1 --loss_selection vgg .8 --loss_selection gan .1 --loss_selection id 2 --loss_selection tv 1 \
--workers 4 \
--pretrain /home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl \
--w_type w+ \
--mode AE_Platon --mbstd-group 2 \
--lr_g 0.0001 --lr_d 0.00005 --lr_e 0.0001 \
--neural_rendering_resolution_initial 128
