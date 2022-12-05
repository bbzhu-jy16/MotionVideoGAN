python -W ignore train.py --name ucf101 --time_step 1 --interpolation_frame_rate 1 --n_frames_G 17 --lr 0.0001 --n_direction 30 --save_direction_path directions/ucf101 --latent_dimension 512 --dataroot datasets/ucf-101 --checkpoints_dir checkpoints/ucf101 --temporal lstm --img_g_weights models/ucf101/network-snapshot-007680.pkl --multiprocessing_distributed --world_size 1 --rank 0 --batchSize 16 --workers 8 --style_gan_size 256 --total_epoch 100 

  
