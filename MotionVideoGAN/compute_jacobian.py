import os
import sys
import time
import argparse
from tqdm import tqdm
import signal
import numpy as np
import torch
from torch.autograd import Variable
import dnnlib
import pickle
import legacy
from torch_utils import training_stats
from torch_utils import custom_ops
from PIL import Image

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

def parse_args():
    signal.signal(signal.SIGINT, lambda x,y: sys.exit(0))

    parser = argparse.ArgumentParser()

    parser.add_argument('--restore_path', type=str, default='outputs/models/faceforensics/network-snapshot-004800.pkl', help = 'The pre-trained model file')
    parser.add_argument('--image_size', type=int, default=256, help = 'size of images')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save the results. If not specified,'
                             '`./outputs/jacobian_seed_{}` will be used '
                             'by default.')
    parser.add_argument('--total_num', type=int, default=10,help='Number of latent codes to sample')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda')
    assert os.path.exists(args.restore_path)
    with dnnlib.util.open_url(args.restore_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    latent_z = Variable(torch.randn([args.total_num, G.z_dim]), requires_grad=True).to(device)

    def f(x): 
        frame_1 = torch.mean(G.synthesis(x.repeat(1,14,1))[:,0:3],dim=1,keepdims=True)
        frame_2 = torch.mean(G.synthesis(x.repeat(1,14,1))[:,3:6],dim=1,keepdims=True)
        
        return torch.cat((frame_1,frame_2),1)

    for num in tqdm(range(args.total_num)):
        latent = G.mapping(latent_z[num:num+1],None,truncation_psi = 1, truncation_cutoff = None, update_emas = False)
        np.save(args.output_dir+'latent'+str(num)+'.npy',latent.detach().cpu().numpy())
        
        jaco_w = torch.autograd.functional.jacobian(f, latent[:,0])
        np.save(args.output_dir+'jacobian'+str(num)+'.npy',jaco_w.detach().cpu().numpy())
        #print(jaco_w.shape)

if __name__=='__main__':
    main()



    

