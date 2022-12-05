import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim 

class CausalConv1D(nn.Conv1d):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1,
                groups=1,
                bias=True):
        super(CausalConv1D, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size-1)*dilation

    def forward(self, input):
        return super(CausalConv1D, self).forward(F.pad(input, (self.__padding,0)))

class code_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,k=5):
        super(code_embedding,self).__init__()
        self.causal_convolution = CausalConv1D(in_channels,embedding_size,kernel_size=k)
    
    def forward(self,x):
        x=self.causal_convolution(x)
        return F.tanh(x)
        
#LSTM
class LSTMModule(nn.Module):
    def __init__(self, 
                forward_direction_path,
                backward_direction_path,
                h_dim=512,
                n_direction=10,
                z_dim=512,
                w_residual=0.2):
        super(RNNModule, self).__init__()
        forward_direction = np.load(forward_direction_path)
        forward_direction = forward_direction[0:n_direction]
        backward_direction = np.load(backward_direction_path)
        backward_direction = backward_direction[0:n_direction]
        self.forward_direction = torch.tensor(forward_direction,dtype=torch.float32)
        self.backward_direction = torch.tensor(backward_direction,dtype=torch.float32)

        self.z_dim=z_dim
        self.h_dim=h_dim
        self.w_residual=w_residual

        self.enc_cell = nn.LSTMCell(z_dim, h_dim)
        self.cell = nn.LSTMCell(z_dim, h_dim)
        self.w = nn.Parameter(torch.FloatTensor(h_dim, n_direction))
        self.b = nn.Parameter(torch.FloatTensor(n_direction))
        self.fc1 = nn.Linear(h_dim*2,z_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(z_dim,z_dim)

        self.init_weights()

    def init_optim(self,lr,beta1,beta2):
        self.optim = optim.Adam(params = self.parameters(),
                                lr=lr,
                                betas=(beta1,beta2),
                                weight_decay=0,
                                eps=1e-8)
    
    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.LSTMCell)):
                for name, param in module.named_parameters():
                    if ('weight_ih' in name) or ('weight_hh' in name):
                        mul = param.shape[0] // 4
                        for idx in range(4):
                            init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            if (isinstance(module, nn.Linear)):
                init.orthogonal_(module.weight)

        nn.init.normal_(self.w, std=0.02)
        self.b.data.fill_(0.0)

    def forward(self,z,n_frame):
        forward_direction = self.forward_direction.cuda(z.get_device())
        backward_direction = self.backward_direction.cuda(z.get_device())
        
        out = [z]

        h_, c_ = self.enc_cell(z)
        h = [h_]
        c = [c_]
        e = []

        for i in range(n_frame-1):
            e_ = self.get_initial_state_z(z.shape[0])
            h_, c_ = self.cell(e_,(h[-1],c[-1]))
            mul = torch.matmul(h_,self.w) + self.b # trainable matrices
            mul = torch.tanh(mul)
            e.append(e_)
            h.append(h_)
            c.append(c_)
            
            if i%2 == 0 :
                direction = backward_direction #the first frame
            else:
                direction = forward_direction #the second frame
            
            out_ = out[-1] + self.w_residual*torch.matmul(mul,direction)
            out.append(out_)

        out = [item.unsqueeze(1) for item in out]
        out = torch.cat(out, dim=1).view(-1,n_frame,self.z_dim) #(bs,frame,512)

        return out

    def get_initial_state_z(self,batchSize):
        return torch.cuda.FloatTensor(batchSize, self.z_dim).normal_()
