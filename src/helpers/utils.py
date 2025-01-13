from math import e
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
import os, time, datetime
import logging
import itertools

from collections import OrderedDict
from torchvision.utils import save_image
import SimpleITK as sitk

META_FILENAME = "metadata.json"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

def uint32_matrix_to_float32(uint_matrix):
    # Scale the uint32 matrix to the range [0, 1]
    scaled_matrix = uint_matrix / np.iinfo(np.uint32).max
    return scaled_matrix

def float32_matrix_to_uint32(float_matrix):
    # Convert the float32 matrix back to uint32
    uint_matrix = (float_matrix * np.iinfo(np.uint32).max).astype(np.uint32)
    return uint_matrix

def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")

def get_model_device(model):
    """Return the device where the model sits."""
    return next(model.parameters()).device

def make_deterministic(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Don't go fast boi :(    
    np.random.seed(seed)

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        # nn.init.xavier_uniform_(m.weight) 
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    # No need to initialize ChannelNorm2D
                
def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def quick_restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    del checkpt
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pad_factor(input_image, spatial_dims, factor):
    """Pad `input_image` (N,C,H,W) such that H and W are divisible by `factor`."""

    if isinstance(factor, int) is True:
        factor_H = factor
        factor_W = factor_H
    else:
        factor_H, factor_W = factor

    H, W = spatial_dims[0], spatial_dims[1]
    pad_H = (factor_H - (H % factor_H)) % factor_H
    pad_W = (factor_W - (W % factor_W)) % factor_W
    return F.pad(input_image, pad=(0, pad_W, 0, pad_H), mode='reflect')

def get_scheduled_params(param, param_schedule, step_counter, ignore_schedule=False):
    # e.g. schedule = dict(vals=[1., 0.1], steps=[N])
    # reduces param value by a factor of 0.1 after N steps
    if ignore_schedule is False:
        vals, steps = param_schedule['vals'], param_schedule['steps']
        assert(len(vals) == len(steps)+1), f'Mispecified schedule! - {param_schedule}'
        idx = np.where(step_counter < np.array(steps + [step_counter+1]))[0][0]
        param *= vals[idx]
    return param

def update_lr(args, optimizer, itr, logger):
    lr = get_scheduled_params(args.learning_rate, args.lr_schedule, itr)
    for param_group in optimizer.param_groups:
        old_lr = param_group['lr']
        if old_lr != lr:
            logger.info('=============================')
            logger.info(f'Changing learning rate {old_lr} -> {lr}')
            param_group['lr'] = lr

def setup_generic_signature(args, directories, special_info):
    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    if args.name is not None:
        args.new_name = '{}_{}_{}_{}'.format(args.name[:5], args.dataset, special_info, time_signature)
    else:
        args.new_name = '{}_{}_{}'.format(args.dataset, special_info, time_signature)

    # print(f'directories: {directories}, args.name: {args.name}')
    args.snapshot = os.path.join(directories, args.new_name)
    args.checkpoints_save = os.path.join(args.snapshot, 'checkpoints')
    makedirs(args.snapshot)
    makedirs(args.checkpoints_save)

    if args.save_storage:
        args.storage_save = os.path.join(args.snapshot, 'storage')
        makedirs(args.storage_save)
    return args

def save_metadata(metadata, directory='results', filename=META_FILENAME, **kwargs):
    """ Save the metadata of a training directory.
    Parameters
    ----------
    metadata:
        Object to save
    directory: string
        Path to folder where to save model. For example './experiments/runX'.
    kwargs:
        Additional arguments to `json.dump`
    """

    makedirs(directory)
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)  #, **kwargs)

def save_model_classifier(model, optimizer, epoch, device, args, logger, save_metadata=False):
    directory = args.checkpoints_save
    makedirs(directory)
    model.cpu()  # Move model parameters to CPU for consistency when restoring

    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('_') or 'logger' in n))
    timestamp = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now())
    args_d['timestamp'] = timestamp

    if save_metadata:
        metadata = dict(image_dims=args.image_dims, epoch=epoch, steps=model.step_counter)
        metadata.update(args_d)
        
        metadata_path = os.path.join(directory, 'metadata/model_{}_metadata_{}.json'.format(model_name, timestamp))
        makedirs(os.path.join(directory, 'metadata'))
        if not os.path.isfile(metadata_path):
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, sort_keys=True)

    # model_name = args.name     
    model_path = os.path.join(directory, 'epoch{}_idx{}_{}.pt'.format(epoch, model.step_counter, timestamp))
    if os.path.exists(model_path):
        model_path = os.path.join(directory, 'epoch{}_idx{}_{:%Y_%m_%d_%H:%M:%S}.pt'.format(epoch, model.step_counter, datetime.datetime.now()))
    
    save_dict = {'model_state_dict': model.state_dict(),
                'classifier_optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'steps': model.step_counter,
                'args': args_d,
                }

    torch.save(save_dict, f=model_path)
    logger.info('Saved model at Epoch {}, to {}'.format(epoch, model_path))
    
    model.to(device)  # Move back to device
    return model_path

def load_optimizer_classifier(args, model, checkpoint, prediction=True):
    if prediction is True:
        model.eval()
        optimizer = None
    else:
        # optimizer
        if args.optim_type=='adam':
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optim_type=='adamw':
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise  NotImplementedError(f'{args.optim_type} not supported yet.')
        optimizer.load_state_dict(checkpoint['classifier_optimizer_state_dict'])
        model.train()
    return model, optimizer

def logger_setup(logpath, filepath, package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())

    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(logpath, mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    # logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger

# Logger and Visdom for Classifier in Step2  
def log_C(epoch, idx, step, mean_epoch_loss, current_loss, best_loss, start_time, epoch_start_time, 
        batch_size, header='[TRAIN]', logger=None, vis=None, title='classification', **kwargs):
    
    improved = ''
    t0 = epoch_start_time
    
    if current_loss < best_loss:
        best_loss = current_loss
        improved = '[*]'  

    # Tensorboard
    if vis is not None:
        if header=='[TRAIN]':
            vis.line(Y=np.array([current_loss]), X=np.array([step]), win='cls', name='train', update=None if step==1 else 'append', opts=dict(title=title, showlegend=True))
        else:
            vis.line(Y=np.array([current_loss]), X=np.array([step]), win='cls', name='valid', update='append', opts=dict(title=title, showlegend=True))
            
    if logger is not None:
        report_f = logger.info   
    else:
        report_f = print

    report_f(f'================>>>{header}, Epoch: {epoch}, Step: {step}')
    if header == '[TRAIN]':
        # report_f(model.args.snapshot)
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Current loss: {:.3f} |"
                 "Rate: {} examples/s | Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, current_loss,
                 int(batch_size*idx / ((time.time()-t0))), time.time()-start_time, improved))
    else:
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Current loss: {:.3f} | Improved: {}".format(epoch, mean_epoch_loss, current_loss, improved))
    return best_loss

def vis_images(vis, real, predict, n_channel, fname, clamp=True):
    # print(f'real[ {real.min()}, {real.max()} ]') #[0,1]
    # print(f'decoded[ {predict.min()}, {predict.max()} ]') #[-0.9,1.1]

    # if args.normalize is True:
    #     # [-1.,1.] -> [0.,1.]
    #     reconstruction = (reconstruction + 1.) / 2.  
    if clamp:
        predict = torch.clamp(predict, min=0., max=1.)
    
    if n_channel==3:
        real = real[:,1:2]
        predict = predict[:,1:2]

    if real.shape[0]>8: # select first 8 images
        real = real[:8]
        predict = predict[:8]
        fname = fname[:8]

    vis.images(real, win='real', opts=dict(title='real', nrow=4))#(batch, 1, width, height)
    vis.images(predict, win='reco', opts=dict(title='reconstruction', nrow=4))
    
    if fname:
        fname_list = [f'{item} \n' for item in fname]
        vis.text(fname_list, win='fname', opts=dict(title='fname'))

    # if epoch == 0 or epoch==args.n_epochs - 2:
    #     fname=os.path.join(args.figures_save, fname)
    #     imgs = torch.cat((real[:,slice:slice+1],decoded[:,slice:slice+1]), dim=0)#torch.Size(batch, 1, width, height)
    #     save_image(imgs, fname, nrow=4, normalize=True, scale_each=True)  

def save_images_utility(args, epoch, vis, input, label, pred, fname):
    # print(f'input[ {input.min()}, {input.max()} ]') #[0,1]
    # print(f'pred[ {pred.min()}, {pred.max()} ]') #[-0.9,1.1]
    slice = input.shape[1]//2

    if args.dataset == 'LiTS':
        # windowing for dispaly
        input = input*float(4095) - 1024
        input = (input + 60) / float(200)
        input[input < 0] = 0
        input[input > 1] = 1

    pred = torch.unsqueeze(pred, axis=1)
    # print('##input, pred###', input.shape, pred.shape, torch.unsqueeze(pred, axis=1).shape)
    vis.images(input[:,slice:slice+1], win='input', opts=dict(title='input', nrow=4))#(batch, 1, width, height)
    vis.images(label[:,slice:slice+1], win='label', opts=dict(title='label', nrow=4))#(batch, 1, width, height)
    vis.images(pred, win='pred', opts=dict(title=fname, nrow=4))#(batch, 1, width, height)

    # if epoch == 0 or epoch==args.n_epochs - 2:
    #     fname=os.path.join(args.figures_save, fname)
    #     imgs = torch.cat((input[:,slice:slice+1],label[:,slice:slice+1],pred), dim=0)#torch.Size(batch, 1, width, height)
    #     save_image(imgs, fname, nrow=4, normalize=True, scale_each=True)  
