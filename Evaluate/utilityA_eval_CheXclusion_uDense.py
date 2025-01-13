"""
Train on real test on anoymized dataset

Adapt DenseNet121 for Utility Classification
CheXclusion is the SOTA on MIMIC-CXR

datasets_utility.MIMIC_CheXclusion
  >Read RGB image
  >Transform

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
"""
import numpy as np
import os, time, datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (MultilabelAccuracy, 
                                        MultilabelAveragePrecision,
                                        MultilabelAUROC,
                                        MultilabelF1Score, 
                                        # MultilabelROC
                                        )
# Custom modules
import sys
sys.path.append('/home/huili/Projects/GAE1/')
from Step4_Evaluate.utilityA_eval_config_CheXclusion_uDense import utility_args, Setttings
from src.network.densenet_utility import DenseNet121
from src.helpers import utils
from src.helpers.datasets import MIMICA as MIMIC

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

def test(args, test_loader, model, device, logger):

    # Metrics statement 
    Accuracy = MultilabelAccuracy(args.num_classes, threshold=0.5, average='macro').to(device)
    AveragePrecision = MultilabelAveragePrecision(args.num_classes, average='macro').to(device)
    AUROC = MultilabelAUROC(args.num_classes, average='macro').to(device)
    F1Score = MultilabelF1Score(args.num_classes, threshold=0.5,average='macro').to(device)
    # ROC = MultilabelROC(args.num_classes).to(device)

    model.eval()  
    preds_tensor = torch.zeros(args.n_data, args.num_classes).to(device, dtype=torch.float32)
    target_tensor = torch.zeros(args.n_data, args.num_classes).to(device, dtype=torch.int)
    left=0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_loader, desc='Test'), 0):
            test_image, test_labels = data
            test_image = test_image.to(device, dtype=torch.float32)
            test_labels = test_labels.to(device, dtype=torch.int)
            y_pred = model(test_image, mode='test')
            # Add sigmoid layer to y_pred: [N, n_class]
            preds = torch.sigmoid(y_pred)
            # Add argmax to labels: [N, n_class]
            target = test_labels
            
            right = left + args.batch_size
            if right>args.n_data:
                print(f'right:{right}')
                right = args.n_data
            preds_tensor[left: right] = preds
            target_tensor[left: right] = target
            left = right  

    # metrics
    criteria = {
        'Accuracy': f'{Accuracy(preds_tensor, target_tensor).item():.3f}',
        'AveragePrecision': f'{AveragePrecision(preds_tensor, target_tensor).item():.3f}',
        'AUROC': f'{AUROC(preds_tensor, target_tensor).item():.3f}',
        'F1Score': f'{F1Score(preds_tensor, target_tensor).item():.3f}',
        }
    print(criteria)

    # Convert and write JSON object to file
    import json
    with open(os.path.join(args.snapshot, "criteria.json"), "w") as outfile: 
        json.dump(criteria, outfile)
        
if __name__ == '__main__':
    # Load arguments from config file
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d = dictify(utility_args)
    args = utils.Struct(**args_d)

    # Set output directories    
    special_info='Utility_'+args.model
    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    if args.name is not None:
        args.name = '{}_{}_{}_{}'.format(args.name[:5], args.dataset, special_info, time_signature)
    else:
        args.name = '{}_{}_{}'.format(args.dataset, special_info, time_signature)
    
    # print(f'directories: {directories}, args.name: {args.name}')
    args.snapshot = os.path.join(Setttings.exp_dir, args.name)
    utils.makedirs(args.snapshot)

    start_time = time.time()
    # Reproducibility
    utils.make_deterministic()
    ####################################################################################################################
    ##                                                                                                                ##
    ##                                                    [ CUDA ]                                                    ##
    ##                                                                                                                ##
    ####################################################################################################################
    # format: model=[Model, N]    
    logger = utils.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))
    
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
    logger.info(f'anony_root: {args.anony_root}')
    logger.info('Input Dimensions: {} {}'.format(args.batch_size, args.image_dims))
    # logger.info(f'normalize: {normalize}')

    testSet = globals()[args.dataset](csv_path=args.csv_path, 
                                image_root=args.anony_root,
                                mode='test',
                                input_nc=args.input_nc,
                                use_transform = args.use_transform,
                                normalize = args.normalize,
                                class_index=-1,
                                use_softlabel=args.use_softlabel,
                                train_cols = args.train_cols,
                                shuffle=False,
                                oui = args.oui)
    
    # collate_fn=multimodal_collate_fn
    test_loader =  DataLoader(testSet, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                               collate_fn=None, pin_memory=True, drop_last=False)
    args.n_data = len(test_loader.dataset)
    logger.info('Evaluation elements: {}'.format(args.n_data))

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                 [ Model Loading]                                                               ##
    ##                                                                                                                ##
    ####################################################################################################################
    assert args.warmstart_ckpt is not None, 'Must provide checkpoint to previously trained AE/HP model.'
    start_time = time.time()

    checkpoint = torch.load(args.warmstart_ckpt)
    args = updata_args(checkpoint, current_args_d=dictify(args))
    model = DenseNet121(num_in_features=args.image_dims[0], pretrained=False, 
                        activations=args.activations, last_activation=args.last_activation, 
                        num_classes=args.num_classes)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)# `strict` False if warmstarting
    del checkpoint
    model = model.to(device)
    
    logger.info(f'Loading model {(time.time() - start_time):.3f}s')
    logger.info("Number of trainable parameters: {}".format(utils.count_parameters(model)))
    logger.info("Estimated model size (under fp32): {:.3f} MB".format(utils.count_parameters(model) * 4. / 10**6))

    ####################################################################################################################
    ##                                                                                                                ##
    ##                                             Utility Classifer Evaluation                                         ##
    ##                                                                                                                ##
    ####################################################################################################################
    test(args, test_loader, model, device, logger)

    logger.info('Time {:.3f} min'.format((time.time() - start_time) / 60))
    logger.info(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))