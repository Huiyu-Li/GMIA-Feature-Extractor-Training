import os
import logging
import time
import torch
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
from visdom import Visdom
from torch.distributed import init_process_group, destroy_process_group

# Custom modules
import sys
sys.path.append('/home/huili/Projects/GAE1/')
from Step2_Model_Pretrain.identity_config import identity_args, Setttings
from src.helpers import utils
from src.network.iresnet import iresnet50, iresnet100
from src.helpers.datasets_identity import get_dataloader
from src.loss.combined_margin import CombinedMarginLoss
from src.loss.partial_fc_v2 import PartialFC_V2
from src.lr_scheduler.lr_scheduler import PolynomialLRWarmup
from src.helpers.utils_distributed_sampler import setup_seed
from src.helpers.utils_distributed_callbacks import CallBackVerification
from src.helpers.utils_distributed_misc import (setup_generic_signature, save_model, 
                                                log_C, AverageMeter)

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

assert torch.__version__ >= "1.12.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.12.0. torch before than 1.12.0 may not work in the future."

def main():
    torch.backends.cudnn.benchmark = True # 在网络固定的情况下，加速训练，一般加在开头
    
    # Get default arguments from config file
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d = dictify(identity_args)
    args = utils.Struct(**args_d)

    # global control random seed
    setup_seed(seed=args.seed, cuda_deterministic=False)

    # Process group initialization
    init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    # Set ouput directories, init_logging, Visdom
    setup_generic_signature(args, Setttings.exp_dir, rank, special_info='Identity_'+args.network)
    args.vis_info = args.vis_info+' '+args.network
    logging.info(f'vis_info: {args.vis_info}')
    vis = (Visdom(env=args.vis_info, server=Setttings.vis_server, port=Setttings.vis_port) if rank == 0 else None)
    # Print args
    # for key, value in args.items():
    #     num_space = 25 - len(key)
    #     logging.info(": " + key + " " * num_space + str(value))
    ##############################################################################################
    ##                                     [ Data Loader ]                                      ##
    ############################################################################################## 
    train_loader, num_image = get_dataloader(args, local_rank)

    ##############################################################################################
    ##                                     [ Model ]                                            ##
    ############################################################################################## 
    # More model choice refer to My_ArcFace foler
    if args.network == "r50":
        backbone = iresnet50(in_channel=args.image_dims[0], dropout=0.0, num_features=args.embedding_size)
    elif args.network == "r100":
        backbone = iresnet100(in_channel=args.image_dims[0], dropout=0.0, num_features=args.embedding_size)
    backbone = backbone.to(local_rank)
    backbone = torch.nn.parallel.DistributedDataParallel(module=backbone, broadcast_buffers=False, 
               device_ids=[local_rank], bucket_cap_mb=16, find_unused_parameters=False)
    backbone.register_comm_hook(None, fp16_compress_hook)
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()
    ############################################################################################## 
    ##                                     [ Loss ]                                             ##
    ############################################################################################## 
    margin_loss = CombinedMarginLoss(64,
                    args.margin_list[0], args.margin_list[1], args.margin_list[2],
                    args.interclass_filtering_threshold)
    module_partial_fc = PartialFC_V2(margin_loss, args.embedding_size, args.num_classes,
                        args.sample_rate, False)
    module_partial_fc = module_partial_fc.to(local_rank)
    # module_partial_fc.train().cuda()
    ##############################################################################################
    ##                                     [ Optimizer ]                                        ##
    ############################################################################################## 
    if args.optim_type == "sgd":
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim_type == "adamw":
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise
    ##############################################################################################
    ##                                     [ Scheduler ]                                        ##
    ############################################################################################## 
    args.total_batch_size = args.batch_size * world_size
    args.warmup_step = num_image // args.total_batch_size * args.warmup_epoch
    args.total_step = num_image // args.total_batch_size * args.n_epochs
    logging.info(f'warmup_step:{args.warmup_step}, total_step: {args.total_step}')
    lr_scheduler = PolynomialLRWarmup(optimizer=opt,
                warmup_iters=args.warmup_step, total_iters=args.total_step)
    
    ##############################################################################################
    ##                                     [ Load Snapshot ]                                    ##
    ############################################################################################## 
    start_epoch = 0
    global_step = 0
    if args.resume:
        dict_checkpoint = torch.load(args.model_path)
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint
    ##############################################################################################
    ##                                     [ CallBacks ]                                        ##
    ############################################################################################## 
    logging.info(f'transform[hflip, rotate]={args.transform}')
    callback_verification = CallBackVerification(args, writer=vis)
        
    # callback_logging = CallBackLogging(frequent=args.frequent,
    #                                     total_step=args.total_step,
    #                                     batch_size=args.batch_size,
    #                                     start_step = global_step,
    #                                     writer=vis)
    ############################################################################################## 
    ##                                     [ Training ]                                         ##
    ############################################################################################## 
    start_time1 = time.time()

    loss_am = AverageMeter()
    amp = torch.cuda.amp.GradScaler(growth_interval=100)# Automatic Mixed Precision package
    backbone.train()
    module_partial_fc.train()
    for epoch in range(start_epoch, args.n_epochs):
        epoch_start_time = time.time()
        train_loader.sampler.set_epoch(epoch)
        ################## Training Start ##################
        for idx, (img, local_labels) in enumerate(train_loader):
            global_step += 1
            local_embeddings = backbone(img)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if args.fp16:
                amp.scale(loss).backward()
                if global_step % args.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
            else:
                loss.backward()
                if global_step % args.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
            lr_scheduler.step()
            with torch.no_grad():
                loss_am.update(loss.item(), 1)
        ################## Training End ##################  
        if rank == 0:
            learning_rate = lr_scheduler.get_last_lr()[0]
            log_C(args, vis, idx, epoch, global_step, loss, loss_am, amp, 
                    learning_rate, world_size, start_time1, epoch_start_time)

        # if global_step % cfg.verbose == 0 and global_step > 0:
            callback_verification(backbone, epoch)

        # if rank == 0:
            args.model_path = save_model(args, epoch, global_step, backbone, module_partial_fc, opt, lr_scheduler)
            logging.info('Saved model at Epoch {}, to {}'.format(epoch, args.model_path))
                
        if args.fp16:
            train_loader.reset()

    if rank == 0:
        args.model_path = save_model(args, epoch, global_step, backbone, module_partial_fc, opt, lr_scheduler)
    destroy_process_group()

if __name__ == "__main__":
    start_time0 = time.time()
    main()
    logging.info('Time {:.3f} min'.format((time.time() - start_time0) / 60))
    logging.info(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))

    # OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 --nnodes=1 ./Step2_Model_Pretrain/identity_classification.py
    # Debug: torchrun --standalone --nnodes=1 --nproc_per_node=1 ./Step2_Model_Pretrain/identity_classification.py
