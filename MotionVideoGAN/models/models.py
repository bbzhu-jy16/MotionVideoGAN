import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from .temporal import LSTMModule
from .model import VideoGenerator
import dnnlib
import legacy

def load_checkpoints(path, gpu):
    if gpu is None:
        ckpt = torch.load(path)
    else:
        loc = 'cuda:{}'.format(gpu)
        ckpt = torch.load(path, map_location=loc)
    return ckpt

def model_to_gpu(model, opt):
    if opt.isTrain:
        if opt.gpu is not None:
            model.cuda(opt.gpu)
            model = DDP(model,
                        device_ids=[opt.gpu],
                        find_unused_parameters=True)
        else:
            model.cuda()
            model = DDP(model, find_unused_parameters=True)
    else:
        model.cuda()
        model = nn.DataParallel(model)

    return model

def create_model(opt):
    assert os.path.exists(opt.img_g_weights)
    with dnnlib.util.open_url(opt.img_g_weights) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(opt.gpu)
    G.eval()
    for p in G.parameters():
        p.requires_grad=False
    
    forward_direction_path = os.path.join(opt.save_direction_path,'forward_direction.npy')
    backward_direction_path = os.path.join(opt.save_direction_path,'backward_direction.npy')
    if opt.temporal=='lstm':
        modelR = LSTMModule(forward_direction_path,
                            backward_direction_path,
                            h_dim = G.z_dim,
                            n_direction=opt.n_direction,
                            z_dim = G.z_dim,
                            w_residual = opt.w_residual)
    else:
        raise NotImplementedError("Not Implemented Yet.")

    video_generator = VideoGenerator(opt.style_gan_size, G.z_dim, G, modelR, opt.interpolation_frame_rate)

    if opt.isTrain:
        from .D_3d import ModelD_3d
        modelR.init_optim(opt.lr,opt.beta1,opt.beta2)
        modelD_3d = ModelD_3d(opt)
        
        modelD_3d_R = ModelD_3d(opt)

        video_generator = model_to_gpu(video_generator,opt)
        modelD_3d = model_to_gpu(modelD_3d,opt)
        modelD_3d_R = model_to_gpu(modelD_3d_R,opt)

        if opt.load_pretrain_path != 'None' and opt.load_pretrain_epoch > -1:
            opt.checkpoints_dir = opt.load_pretrain_path
            m_name = '/modelR_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            video_generator.module.modelR.load_state_dict(ckpt)

            m_name = '/modelD_3d_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            modelD_3d.load_state_dict(ckpt)

            m_name = '/modelD_3d_R_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            modelD_3d_R.load_state_dict(ckpt)

        return [video_generator, modelD_3d, modelD_3d_R]
    else:
        modelR.eval()
        for p in modelR.parameters():
            p.requires_grad = False
        video_generator.modelR = modelR
        video_generator = model_to_gpu(video_generator, opt)

        if opt.load_pretrain_path != 'None' and opt.load_pretrain_epoch > -1:
            m_name = '/modelR_epoch_%d.pth' % (opt.load_pretrain_epoch)
            ckpt = load_checkpoints(opt.load_pretrain_path + m_name, opt.gpu)
            video_generator.module.modelR.load_state_dict(ckpt,strict=False)
        return video_generator
