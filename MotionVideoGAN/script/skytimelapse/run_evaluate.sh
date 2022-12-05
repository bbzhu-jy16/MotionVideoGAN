python -W ignore evaluate.py --save_direction_path directions/skytimelapse --latent_dimension 512 --style_gan_size 256 --img_g_weights models/sky_timelapse/network-snapshot-008600.pkl --n_direction 30 --temporal lstm --load_pretrain_path checkpoints/sky_timelapse/ --load_pretrain_epoch 100 --results results/sky_timelapse --num_test_videos 2048 --n_frames_G 17 --w_residual=0.2

