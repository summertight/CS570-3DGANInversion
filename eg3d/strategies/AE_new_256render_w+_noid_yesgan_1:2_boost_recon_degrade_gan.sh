echo 'AE_new train start'

CUDA_VISIBLE_DEVICES=0,2,4 \
python train.py --outdir=./ae_new_128_w+_Platon_boost_recon_degrage_gan \
--gpus=3 --batch 12 --gamma 1 \
--loss_selection l1 10 --loss_selection vgg 10 --loss_selection gan .01 --loss_selection id 1 --loss_selection tv 1 \
--workers 4 \
--pretrain /home/nas4_user/jaeseonglee/ICCV2023/eg3d_ckpts/eg3d-fixed-triplanes-ffhq.pkl \
--w_type w+ \
--mode AE_Platon --mbstd-group 4 \
--lr_g 0.0001 --lr_d 0.00005 --lr_e 0.0001 \
--neural_rendering_resolution_initial 128
