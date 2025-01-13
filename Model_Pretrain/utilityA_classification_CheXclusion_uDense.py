"""
Adapt DenseNet121 for Utility Classification
CheXclusion is the SOTA on MIMIC-CXR

datasets.MIMIC
  >input_nc=3
  >normalize=True

utility_config_CheXclusion_uDense.py
  >batch_size=48
  >image_dims = (3,512,512)
  >pretrained=True

Model: src.network.densenet_utility
  self.classifier = nn.Sequential(OrderedDict([
      ('fc0', nn.Linear(num_features, 512)),
      ('elu0', self.activation(inplace=True)),
      ('fc1', nn.Linear(512, num_classes))
      ]))
  state_dict.pop('classifier.fc0.weight', None)
  state_dict.pop('classifier.fc0.bias', None)
  state_dict.pop('classifier.fc1.weight', None)
  state_dict.pop('classifier.fc1.bias', None)   

Main()
  from src.network.densenet_utility import DenseNet121
  Model: DenseNet121(initialized with pre-trained weights from ImageNet27)
  Loss: multi-label BCELoss
  Optimizer: Adam, best initial LR: 0.0005 for CXR and NIH, 0.0001 for CXP. 
  scheduler: scheduler.step(epoch_valid_loss[-1])
"""
import numpy as np
import os, time
from tqdm import tqdm
import argparse
import torch
from visdom import Visdom
from torch.utils.data import DataLoader

# Custom modules
import sys
sys.path.append('/home/huili/Projects/GAE1/')
from Step2_Model_Pretrain.utilityA_config_CheXclusion_uDense import utility_args, Setttings
from src.network.densenet_utility import DenseNet121
# from src.loss.classification import BCEwithClassWeights
from src.helpers import utils
from src.helpers.datasets import MIMICA as MIMIC  # MIMICA to read x_A.jpg

# go fast boi!!
torch.backends.cudnn.benchmark = True

def updata_args(checkpoint, current_args_d):
    loaded_args_d = checkpoint['args']
    args = utils.Struct(**loaded_args_d)

    if current_args_d is not None:
        for k,v in current_args_d.items():
            try:
                loaded_v = loaded_args_d[k]
            except KeyError:
                logger.warning('Argument {} (value {}) not present in recorded arguments. Using current argument.'.format(k,v))
                continue
            # if loaded_v !=v:
                # logger.warning('Current argument {} (value {}) does not match recorded argument (value {}). Recorded argument will be overriden.'.format(k, v, loaded_v))
        # HACK
        loaded_args_d.update(current_args_d)
        args = utils.Struct(**loaded_args_d)
    return args

def optimize_loss(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def valid(args, model, loss_fn, epoch, idx, train_image, train_labels, valid_image, valid_labels, device, 
    epoch_valid_loss, best_valid_loss, start_time, epoch_start_time, logger, vis):
    
    model.eval()  
    with torch.no_grad():
        valid_image = valid_image.to(device, dtype=torch.float32)
        valid_labels = valid_labels.to(device, dtype=torch.float32)
        y_pred = model(valid_image, mode='valid')
        valid_loss = loss_fn(y_pred, valid_labels)

        epoch_valid_loss.append(valid_loss.item())
        mean_valid_loss = np.mean(epoch_valid_loss)
        
        best_valid_loss = utils.log_C(epoch, idx, model.step_counter, mean_valid_loss, valid_loss.item(), 
                                     best_valid_loss, start_time, epoch_start_time, 
                                     batch_size=valid_image.shape[0], header='[VALID]', 
                                     logger=logger, vis=vis, title = 'Utility Classification')
   
    return best_valid_loss, epoch_valid_loss

def train(args, model, loss_fn, train_loader, valid_loader, device, logger, optimizer):
    start_time = time.time()

    vis = Visdom(env=args.vis_info, server=Setttings.vis_server, port=Setttings.vis_port)

    valid_loader_iter = iter(valid_loader)
    best_loss, best_valid_loss, mean_epoch_loss = np.inf, np.inf, np.inf
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for epoch in range(args.start_epoch, args.n_epochs):
        logger.info('Epoch: {}'.format(epoch))
        epoch_start_time = time.time()
        epoch_loss, epoch_valid_loss = [], []         
        
        model.train()
        for idx, data in enumerate(tqdm(train_loader, desc='Train'), 0):
            train_image, train_labels = data
            train_image = train_image.to(device, dtype=torch.float32)
            train_labels = train_labels.to(device, dtype=torch.float32)
            y_pred = model(train_image, mode='train')
            train_loss = loss_fn(y_pred, train_labels)
            optimize_loss(train_loss, optimizer)
            
            if model.step_counter % args.log_interval == 1:
                print(f'epoch: {epoch}, step: {model.step_counter}')
                epoch_loss.append(train_loss.item())
                mean_epoch_loss = np.mean(epoch_loss)
                
                best_loss = utils.log_C(epoch, idx, model.step_counter, mean_epoch_loss, train_loss.item(),
                                best_loss, start_time, epoch_start_time, batch_size=train_image.shape[0],
                                logger=logger, vis = vis, title = 'Utility Classification')            
                try:
                    valid_image, valid_labels = next(valid_loader_iter)
                except StopIteration:
                    valid_loader_iter = iter(valid_loader)
                    valid_image, valid_labels = next(valid_loader_iter)

                best_valid_loss, epoch_valid_loss = valid(args, model, loss_fn, epoch, idx, train_image, train_labels, valid_image, valid_labels, 
                    device, epoch_valid_loss, best_valid_loss, start_time, epoch_start_time, logger, vis)

                model.train()

                # LR scheduling
                print(f'len(epoch_valid_loss)={len(epoch_valid_loss)}')
                scheduler.step(epoch_valid_loss[-1])
                        
        ckpt_path = utils.save_model_classifier(model, optimizer, epoch, device, args=args, logger=logger)

        # End epoch
        mean_epoch_loss = np.mean(epoch_loss)
        mean_epoch_valid_loss = np.mean(epoch_valid_loss)

        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean valid loss: {:.3f}'.format(epoch, 
            mean_epoch_loss, mean_epoch_valid_loss))    

    logger.info("Training complete. Time elapsed: {:.3f} s. Number of steps: {}".format((time.time()-start_time), epoch))

if __name__ == '__main__':
    description = "Utility Classification Pre-training."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # General options - see `utility_config.py` for full options
    general = parser.add_argument_group('General options')
    general.add_argument('--model', type=str, default='DenseNet121', help='the model name')
    general.add_argument('--start_epoch', type=int, default=0, help="The start epochs for continue training")
    general.add_argument('--data_train', type=str, default=None, help='the training image root')
    general.add_argument('--data_valid', type=str, default=None, help='the validation image root')
    general.add_argument('--vis', type=str, default=None, help='Visdom window')

    # Warmstart training
    warmstart_args = parser.add_argument_group("Warmstart options")
    warmstart_args.add_argument("-warmstart", "--warmstart", help="Warmstart training from ckpt.", action="store_true")
    warmstart_args.add_argument("-ckpt", "--warmstart_ckpt", default=None, help="Path to ckpt.")

    cmd_args = parser.parse_args()

    # Override default arguments from config file with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(utility_args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = utils.Struct(**args_d)
    # Set ouput directories
    args = utils.setup_generic_signature(args, Setttings.exp_dir, special_info='Utility_'+args.model)
    
    start_time = time.time()
    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    # format: model=[Model, N]    
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))
    args.vis_info = args.name+'_'+ args.vis
    logger.info(f'vis_info: {args.vis_info}')
    
    if torch.cuda.is_available():
        logger.info('Using GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
        device = torch.device("cuda")
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        logger.info('Using CPU')
        device = torch.device("cpu")
        # torch.set_default_tensor_type('torch.FloatTensor')
    logger.info(f'Device: {device}, GPU ID: {args.gpu_ids}')
    logger.info(f'Save Dir: {args.snapshot}')
    
    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                [ Data Loader ]                                                 ##
    ##                                                                                                                ##
    ####################################################################################################################
    logger.info('Loading dataset: {}'.format(args.dataset))
    logger.info(f'csv_path: {args.csv_path}')
    logger.info(f'train data: {args.data_train}')
    logger.info(f'valid data: {args.data_valid}')
    logger.info('Input Dimensions: {} {}'.format(args.batch_size, args.image_dims))
    # logger.info(f'normalize: {normalize}')

    trainSet = globals()[args.dataset](csv_path=args.csv_path, 
                                  image_root=args.data_train,
                                  mode='train',
                                  input_nc=args.input_nc,
                                  use_transform = args.use_transform,
                                  normalize = args.normalize,
                                  class_index=-1,
                                  use_softlabel=args.use_softlabel,
                                  train_cols = args.train_cols,
                                  shuffle=True,
                                  oui = args.oui)
    validSet = globals()[args.dataset](csv_path=args.csv_path, 
                                  image_root=args.data_valid,
                                  mode='valid',
                                  input_nc = 3,
                                  use_transform = args.use_transform,
                                  normalize = args.normalize,
                                  class_index=-1,
                                  use_softlabel=args.use_softlabel,
                                  train_cols = args.train_cols,
                                  shuffle=False,
                                  oui = args.oui)
    
    n_instance, class_instance_nums = trainSet.get_class_instance_num()
    logger.info(f'N_instance: {n_instance}')
    logger.info(f'Class_instance_nums: {class_instance_nums}')
    
    # collate_fn=multimodal_collate_fn
    train_loader =  DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               collate_fn=None, pin_memory=True, drop_last=True)
    valid_loader =  DataLoader(validSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                               collate_fn=None, pin_memory=True, drop_last=False)
    
    # args.log_interval = math.ceil(args.n_data/args.batch_size)
    # logger.info('Log Interval: {}'.format(args.log_interval))
    # check imbalance ratio for each task
    # imratio = traindSet.imbalance_ratio(verbose=True)
    # logger.info(f'Imbalance ratio: {imratio}')

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                 [ Resume Training or Training from Scratch ]                                   ##
    ##                                                                                                                ##
    ####################################################################################################################
    # Loss
    # loss_fn = BCEwithClassWeights(class_instance_nums, n_instance) # CheXFusion, weight based on trainSet
    loss_fn = torch.nn.BCEWithLogitsLoss() # TorchXRayVision
    loss_fn = loss_fn.to(device)

    if args.warmstart is True:
        assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
        start_time = time.time()

        checkpoint = torch.load(args.warmstart_ckpt)
        args = updata_args(checkpoint, current_args_d=dictify(args))
        args.start_epoch = checkpoint['epoch'] + 1
        model = DenseNet121(num_in_features=args.image_dims[0], pretrained=False, 
                            activations=args.activations, last_activation=args.last_activation, 
                            num_classes=args.num_classes)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)# `strict` False if warmstarting
        del checkpoint
        model = model.to(device)
        
        logger.info(f'Loading model {(time.time() - start_time):.3f}s')
        # optimizer
        model, optimizer = utils.load_optimizer_classifier(args, model, checkpoint, prediction=False)
    else:
        start_time = time.time()

        model = DenseNet121(num_in_features=args.image_dims[0], pretrained=args.pretrained, 
                            activations=args.activations, last_activation=args.last_activation, 
                            num_classes=args.num_classes)
        model = model.to(device)

        logger.info(f'Initializing model {(time.time() - start_time):.3f}s')
        # optimizer
        if args.optim_type=='adam':
            optimizer = torch.optim.Adam(params=model.parameters(),  lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optim_type=='adamw':
            optimizer= torch.optim.AdamW(params=model.parameters(),  lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise  NotImplementedError(f'{args.optim_type} not supported yet.')    
        
    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated model size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))
    # logger.info('Optimizers: {}'.format(optimizers))
    # metadata = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    # logger.info(metadata)

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                             Utility Classifer Training                                         ##
    ##                                                                                                                ##
    ####################################################################################################################
    train(args, model, loss_fn, train_loader, valid_loader, device, logger, optimizer=optimizer)

    logger.info('Time {:.3f} min'.format((time.time() - start_time) / 60))
    logger.info(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))