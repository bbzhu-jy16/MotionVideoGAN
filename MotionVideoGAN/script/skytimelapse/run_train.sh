python -W ignore train.py --name sky_timelapse --time_step 1 --interpolation_frame_rate 1 --n_frames_G 17 --lr 0.0001 --n_direction 30 --save_direction_path directions/sky_timelapse --latent_dimension 512 --dataroot datasets/sky_timelapse --checkpoints_dir checkpoints/sky_timelapse --temporal lstm --img_g_weights models/skytimelapse/network-snapshot-008600.pkl --multiprocessing_distributed --world_size 1 --rank 0 --batchSize 16 --workers 8 --style_gan_size 256 --total_epoch 100 

  
