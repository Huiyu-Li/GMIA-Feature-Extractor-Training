import importlib
import os, sys
import time, datetime
import torch
import logging
import numpy as np

def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = os.path.basename(config_file)
    temp_module_name = os.path.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = os.path.join('work_dirs', temp_module_name)
    return cfg

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def init_logging(rank, logpath):
    # if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(funcName)s: %(message)s', "%H:%M:%S")
        handler_file = logging.FileHandler(os.path.join(logpath, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)
        log_root.info('Save Dir: %s' % logpath)

def setup_generic_signature(cfg, directories, rank, special_info):
    if rank == 0:
        time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
        if cfg.name is not None:
            cfg.name = '{}_{}_{}_{}'.format(cfg.name[:5], cfg.dataset, special_info, time_signature)
        else:
            cfg.name = '{}_{}_{}'.format(cfg.dataset, special_info, time_signature)
        cfg.output = os.path.join(directories, cfg.name)
        cfg.checkpoints_save = os.path.join(cfg.output, 'checkpoints')
        makedirs(cfg.output)
        makedirs(cfg.checkpoints_save)

        init_logging(rank, cfg.output)

def save_model(cfg, epoch, global_step, backbone, module_partial_fc, opt, lr_scheduler):
    timestamp = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()) 

    if cfg.save_all_states:
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "state_dict_backbone": backbone.module.state_dict(),
            "state_dict_softmax_fc": module_partial_fc.state_dict(),
            "state_optimizer": opt.state_dict(),
            "state_lr_scheduler": lr_scheduler.state_dict()
            }
    else:
        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "state_dict_backbone": backbone.module.state_dict(),
            # "state_dict_softmax_fc": module_partial_fc.state_dict(),
            # "state_optimizer": opt.state_dict(),
            # "state_lr_scheduler": lr_scheduler.state_dict()
            }
    model_path = os.path.join(cfg.checkpoints_save, '{}_epoch{}_idx{}_{}.pt'.format(
                cfg.network, epoch, global_step, timestamp))
    if os.path.exists(model_path):
        model_path = os.path.join(cfg.checkpoints_save, '{}_epoch{}_idx{}_{:%Y_%m_%d_%H:%M:%S}.pt'.format(
                    cfg.network, epoch, global_step, datetime.datetime.now()))
    torch.save(checkpoint, f=model_path)

    return model_path

@torch.no_grad()
def log_C(cfg, writer, idx, epoch, global_step, loss, loss_am, amp, learning_rate, world_size, start_time1, epoch_start_time):
    try:
        speed: float = idx * cfg.batch_size * world_size / (time.time() - epoch_start_time)
        speed_total = speed * world_size
    except ZeroDivisionError:
        speed_total = float('inf')

    time_sec_avg = int(time.time() - start_time1) / (global_step + 1)
    eta_sec = time_sec_avg * (cfg.total_step - global_step - 1)
    time_for_end = eta_sec/3600

    if writer is not None:
        writer.line(Y=np.array([time_for_end]), X=np.array([epoch]), win='time', name='time_', 
                        update=None if epoch==0 else 'append', opts=dict(title='Time for End', showlegend=False))
        writer.line(Y=np.array([learning_rate]), X=np.array([epoch]), win='lr', name='lr_', 
                        update=None if epoch==0  else 'append', opts=dict(title='Learning Rate', showlegend=False))
        writer.line(Y=np.array([loss.item()]), X=np.array([epoch]), win='loss', name='train', 
                        update=None if epoch==0  else 'append', opts=dict(title='Loss', showlegend=True))
    if cfg.fp16:
        msg = "Speed %.2f samples/s | Loss %.4f | LR %.6f | Epoch: %d | Global Step: %d | " \
            "Fp16 Grad Scale: %2.f | Required: %1.f hours" % (speed_total, loss_am.avg, learning_rate, epoch, 
            global_step, amp.get_scale(), time_for_end)
    else:
        msg = "Speed %.2f samples/s | Loss %.4f | LR %.6f | Epoch: %d | Global Step: %d | " \
            "Required: %1.f hours" % (speed_total, loss_am.avg, learning_rate, epoch, global_step, time_for_end)
    logging.info(msg)
    loss_am.reset()

class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
