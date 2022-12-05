python -W ignore train.py --name faceforensics --time_step 1 --interpolation_frame_rate 1 --n_frames_G 17 --lr 0.0001 --n_direction 30 --save_direction_path directions/faceforensics --latent_dimension 512 --dataroot datasets/faceforensics/face256px/train_image --checkpoints_dir checkpoints/faceforensics --img_g_weights models/faceforensics/network-snapshot-004800.pkl --temporal lstm --multiprocessing_distributed --world_size 1 --rank 0 --batchSize 16 --workers 8 --style_gan_size 256 --total_epoch 100

  
