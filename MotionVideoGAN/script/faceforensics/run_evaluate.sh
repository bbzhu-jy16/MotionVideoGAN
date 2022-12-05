python -W ignore evaluate.py --save_direction_path directions/faceforensics --latent_dimension 512 --style_gan_size 256 --img_g_weights models/faceforensics/network-snapshot-004800.pkl --n_direction 30 --temporal lstm --load_pretrain_path checkpoints/faceforensics/  --load_pretrain_epoch 100 --results results/faceforensics --num_test_videos 2048 --n_frames_G 17  --w_residual 0.2


