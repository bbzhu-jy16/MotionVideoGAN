python -W ignore evaluate.py --save_direction_path directions/ucf101 --latent_dimension 512 --style_gan_size 256 --img_g_weights models/ucf101/network-snapshot-007680.pkl --n_direction 30 --temporal lstm --load_pretrain_path checkpoints/ucf101/ --load_pretrain_epoch 100 --results results/ucf101 --num_test_videos 2048 --n_frames_G 17 --w_redisual 0.2