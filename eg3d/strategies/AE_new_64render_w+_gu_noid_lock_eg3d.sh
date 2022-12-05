echo 'AE_new train start LOCK EG3D for warm up'
echo 'This for warm-up'
CUDA_VISIBLE_DEVICES=0,1,2 \
python train.py --outdir=/home/nas2_userF/gyojunggu/gyojung/faceswap/eg3d/eg3d/ae_new_128_w+_1202_noid_lock_eg3d \
--gpus=3 --batch 12 --gamma 1 \
--loss_selection l1 1 --loss_selection vgg 1 \
--workers 4 \
--pretrain /home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl \
--w_type w+ \
--mode AE --mbstd-group 2 \
--lr_g 0.00005 --lr_d 0.00008 --lr_e 0.00005 \
--neural_rendering_resolution_initial 64
