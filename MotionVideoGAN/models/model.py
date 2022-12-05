import math
import random
import torch
from torch import nn
from torch.nn import functional as F

class VideoGenerator(nn.Module):
    def __init__(self,
                size,
                style_dim,
                style_generator,
                modelR=None,
                interpolation_frame_rate=1):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        self.style_generator = style_generator
        self.modelR = modelR
        self.mapping = style_generator.mapping
        self.synthesis = style_generator.synthesis
        self.interpolation_frame_rate=interpolation_frame_rate

    def get_latent(self,input):
        return self.mapping(input,None,truncation_psi = 1, truncation_cutoff = None, update_emas = False)

    def forward(self,
                latent,
                n_frame):
        latent = self.mapping(latent,None,truncation_psi = 1, truncation_cutoff = None, update_emas = False)  
        latent = latent[:,0,:]
        outputs = []
        styles= self.modelR(latent,n_frame)
        styles = styles.unsqueeze(2)
        styles = styles.repeat(1,1,14,1)
        imgs = self.synthesis(styles[:,0])
        outputs.append(imgs[:,3:6])
        for i in range(1,n_frame-1):
            for k in range(self.interpolation_frame_rate):
                latent_code = styles[:,i]*(k)/self.interpolation_frame_rate+styles[:,i-1]*(self.interpolation_frame_rate-k)/self.interpolation_frame_rate
                imgs = self.synthesis(latent_code)               
                if (i-1)%2==0:
                    outputs.append(imgs[:,0:3])
                else:
                    outputs.append(imgs[:,3:6])
        outputs = [item.unsqueeze(1) for item in outputs]
        outputs = torch.cat(outputs,dim=1)

        return outputs

