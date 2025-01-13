import os
import logging
import time, datetime
import torch
import numpy as np
import sklearn
from torch.utils.data import DataLoader
# Custom modules
import sys
sys.path.append('/home/huili/Projects/GAE1/')
from Step4_Evaluate.identity_eval_config import identity_args, Setttings
from src.helpers import utils
from src.network.iresnet import iresnet50, iresnet100
from src.helpers.datasets_identity import MIMIC_EVAL
from src.helpers.utils_distributed_sampler import setup_seed
from src.helpers.utils_distributed_misc import makedirs
from src.helpers.metrics_identity import calculate_f1_score, calculate_val_far

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

def main():
    # Get default arguments from config file
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d = dictify(identity_args)
    args = utils.Struct(**args_d)

    # global control random seed
    setup_seed(seed=args.seed, cuda_deterministic=False)

    # Set ouput directories, init_logging  
    special_info='Identity_'+args.network
    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    if args.name is not None:
        args.name = '{}_{}_{}_{}'.format(args.name[:5], args.dataset, special_info, time_signature)
    else:
        args.name = '{}_{}_{}'.format(args.dataset, special_info, time_signature)
    args.output = os.path.join(Setttings.exp_dir, args.name)
    makedirs(args.output)
    logger = utils.logger_setup(logpath=os.path.join(args.output, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(f'Save Dir: {args.output}')

    # Process group initialization
    if torch.cuda.is_available():
        logger.info('Using GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
        device = torch.device("cuda")
    else:
        logger.info('Using CPU')
        device = torch.device("cpu")
    logger.info(f'Device: {device}, GPU ID: {args.gpu_ids}')

    # logger.info args
    # for key, value in args.items():
    #     num_space = 25 - len(key)
    #     logging.info(": " + key + " " * num_space + str(value))
    ##############################################################################################
    ##                                     [ Data Loader ]                                      ##
    ############################################################################################## 
    logger.info('Loading dataset: {}'.format(args.dataset))
    logger.info(f'csv_path: {args.csv_path}')
    logger.info(f'image_root: {args.image_root}')
    testSet = MIMIC_EVAL(csv_path = args.csv_path,
                    image_root=args.image_root,
                    input_nc=args.input_nc,
                    transform = args.transform,
                    shuffle = args.shuffle, # Need shuffle=FALSE, if already shuffled
                 )

    valid_loader = DataLoader(testSet, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, collate_fn=None, 
                            pin_memory=True, drop_last=False)
    ##############################################################################################
    ##                                     [ Model ]                                            ##
    ############################################################################################## 
    # More model choice refer to My_ArcFace foler
    if args.network == "r50":
        backbone = iresnet50(in_channel=args.image_dims[0], dropout=0.0, num_features=args.embedding_size)
    elif args.network == "r100":
        backbone = iresnet100(in_channel=args.image_dims[0], dropout=0.0, num_features=args.embedding_size)
    backbone = backbone.to(device)
    ##############################################################################################
    ##                                     [ Load Snapshot ]                                    ##
    ############################################################################################## 
    dict_checkpoint = torch.load(args.model_path)
    # dict_keys(['epoch', 'global_step', 'state_dict_backbone', 'state_dict_softmax_fc', 'state_optimizer', 'state_lr_scheduler'])
    backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    del dict_checkpoint
    ############################################################################################## 
    ##                                     [ Evaluation ]                                         ##
    ############################################################################################## 
    backbone.eval()
    num_pairs = testSet.__len__()
    logger.info(f'len(data): {num_pairs}')
    issame_arr = np.zeros((num_pairs))
    dist_arr = np.zeros((num_pairs))
    left=0
    with torch.no_grad():
        for idx, (img1, img2, issame) in enumerate(valid_loader):
            right = left+ args.batch_size
            if right>num_pairs:
                logger.info(f'right:{right}')
                right = num_pairs
            issame_arr[left:right] = issame
            if all(args.transform):
                image1 = img1[0].to(device); image2 = img2[0].to(device)
                hflip1 = img1[1].to(device); hflip2 = img2[1].to(device)
                rotate1 = img1[2].to(device); rotate2 = img2[2].to(device)
                embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2) 
                embeddings_hflip1 = backbone(hflip1); embeddings_hflip2 = backbone(hflip2)
                embeddings_rotate1 = backbone(rotate1); embeddings_rotate2 = backbone(rotate2)
                embeddings1 = embeddings_image1+embeddings_hflip1+embeddings_rotate1
                embeddings2 = embeddings_image2+embeddings_hflip2+embeddings_rotate2
            elif args.transform[0]:
                image1 = img1[0].to(device); image2 = img2[0].to(device)
                hflip1 = img1[1].to(device); hflip2 = img2[1].to(device)
                embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2) 
                embeddings_hflip1 = backbone(hflip1); embeddings_hflip2 = backbone(hflip2)
                embeddings1 = embeddings_image1+embeddings_hflip1
                embeddings2 = embeddings_image2+embeddings_hflip2
            elif args.transform[1]:
                image1 = img1[0].to(device); image2 = img2[0].to(device)
                rotate1 = img1[1].to(device); rotate2 = img2[1].to(device)
                embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2)
                embeddings_rotate1 = backbone(rotate1); embeddings_rotate2 = backbone(rotate2)
                embeddings1 = embeddings_image1+embeddings_rotate1
                embeddings2 = embeddings_image2+embeddings_rotate2
            else:
                image1 = img1.to(device); image2 = img2.to(device)
                embeddings1 = backbone(image1); embeddings2 = backbone(image2)              

            # Scale input vectors individually to unit norm (vector length).
            embeddings1 = sklearn.preprocessing.normalize(embeddings1.detach().cpu().numpy()) # (N, 512)
            embeddings2 = sklearn.preprocessing.normalize(embeddings2.detach().cpu().numpy()) 
            # Check Size
            assert (embeddings1.shape[0] == embeddings2.shape[0])
            assert (embeddings1.shape[1] == embeddings2.shape[1])
            diff = np.subtract(embeddings1, embeddings2) # (N, 512)
            dist = np.sum(np.square(diff), 1) # (N)
            # Check Size np.sqrt(np.sum(embeddings1*embeddings1))
            dist_arr[left:right] = dist
                
            left = right
    
    # Calculate evaluation metrics
    logger.info(f'transform[hflip, rotate]={args.transform}')

    f1_score, precision, recall, acc = calculate_f1_score(args.best_threshold, dist_arr, issame_arr, all=True)
    logger.info(f'F1_score:{f1_score:.3f}, precision:{precision:.3f}, recall:{recall:.3f}, acc:{acc:.3f}')

    tar, far, frr, ACA = calculate_val_far(args.best_threshold, dist_arr, issame_arr)
    logger.info(f'tar:{tar:.3f}, far: {far:.3f}, frr: {frr:.3f}, ACA: {ACA:.3f}')    

if __name__ == "__main__":
    start_time0 = time.time()
    main()
    logging.info('Time {:.3f} min'.format((time.time() - start_time0) / 60))
    logging.info(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))