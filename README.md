# MotionVideoGAN pytorch implementation

## Requirements
* Linux is supported.
* 64-bit Python 3.8 (or later) and PyTorch 1.9.0 (or later).
* CUDA toolkit 11.1 or later. 
* GCC 7.5 or later (Linux) compilers. 
* Python libraries:
  - numpy>=1.20
  - pandas=1.4.1
  - moviepy=1.0.3
  - click>=8.0
  - pillow=8.3.1
  - scipy=1.7.1
  - pytorch=1.9.1
  - torchvision=0.11.3
  - cudatoolkit=11.1
  - requests=2.26.0
  - tqdm=4.62.2
  - ninja=1.10.2
  - matplotlib=3.4.2
  - imageio=2.9.0
  - imgui==1.3.0
  - glfw==2.2.0
  - pyopengl==3.1.5
  - imageio-ffmpeg==0.4.3
  - pyspng
  - tqdm
  - tensorboard


# MotionStyleGAN
We modified the source code of StyleGAN3 (pytorch) to generate two images sharing the same contents but producing different motions. 

## Preparing datasets
Reorganized datasets are needed for MotionStyleGAN training. We recommend concating image pairs and using our code for dataset generation.

### 256x256 resolution source data for example
python dataset_tool.py --source=dataset_path/images_concated --dest=training_data_path/data.zip --resolution=256x512

## Train MotionStyleGAN model
cd MotionStyleGAN

### UCF101 dataset for example
python train.py --outdir=~/training-runs --cfg=stylegan2 --data=training_data_path/data.zip \
    --gpus=4 --batch=32 --gamma=1 --mirror=1 --kimg=25000 --snap=50 

### Fine-tune pre-trained models
python train.py --outdir=~/training-runs --cfg=stylegan2 --data=training_data_path/data.zip \
    --gpus=4 --batch=32 --gamma=1 --mirror=1 --kimg=5000 --snap=5 \
    --resume=~/training-runs/pre-trained_model.pkl


## Motion Code Generation
cd ../MotionVideoGAN

### Compute Jacobian Matrix
python compute_jacobian.py --restore_path ~/training-runs/pre-trained_model.pkl

### Compute Motion Codes
python compute_directions.py jacobian.npy

## Train MotionVideoGAN Model
Our code is repoduced based on MoCoGAN-HD.

### UCF101 for example
bash script/ucf101/run_train.sh

### Generate Videos with MotionVideoGAN Model(UCF101 for example)
bash script/ucf101/run_evaluate.sh

## Acknowledgements
Our code borrows code from StyleGAN3[1], LowRankGAN[2], and MoCoGAN-HD[3].
[1] https://github.com/NVlabs/stylegan3
[2] https://github.com/zhujiapeng/LowRankGAN
[3] https://github.com/snap-research/MoCoGAN-HD