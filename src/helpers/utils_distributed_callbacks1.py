import logging
import sklearn
import numpy as np
from typing import List
import torch
from torch import distributed
from torch.utils.data import DataLoader
# Custom modules
from .datasets_identity1 import MIMIC_EVAL
from .metrics_identity import get_threshold

class CallBackVerification(object):
    
    def __init__(self, args, writer=None):
        self.args = args
        self.rank: int = distributed.get_rank()
        self.highest_f1: float = 0.0
        self.highest_thr: float = 0.0
        self.best_f1_list = []
        self.best_thr_list = []

        # self.thresholds = np.arange(0, 4, 0.01) # 400
        self.thresholds = np.arange(0, 4, 0.001) # 4000
    
        if self.rank == 0:
            logging.info(f'csv_path: {args.csv_valid_path}')
            validSet = MIMIC_EVAL(csv_path = args.csv_valid_path,
                            image_root=args.image_root,
                            input_nc=args.input_nc,
                            transform = args.transform,
                            shuffle = args.shuffle, # Need shuffle=FALSE, if already shuffled
                        )
            self.valid_loader = DataLoader(validSet, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.num_workers, collate_fn=None, 
                                    pin_memory=True, drop_last=False)
            self.num_pairs = validSet.__len__()   
            logging.info(f'len(data): {self.num_pairs}') 
        self.writer = writer

    def ver_test(self, backbone: torch.nn.Module, epoch):
      issame_arr = np.zeros((self.num_pairs))
      dist_arr = np.zeros((self.num_pairs))
      left=0
      device = torch.device("cuda")
      with torch.no_grad():
          for idx, (img1, img2, issame) in enumerate(self.valid_loader):
              right = left+ self.args.batch_size
              if right>self.num_pairs:
                  logging.info(f'right:{right}')
                  right = self.num_pairs
              issame_arr[left:right] = issame
              if all(self.args.transform):
                  image1 = img1[0].to(device); image2 = img2[0].to(device)
                  hflip1 = img1[1].to(device); hflip2 = img2[1].to(device)
                  rotate1 = img1[2].to(device); rotate2 = img2[2].to(device)
                  
                  embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2) 
                  embeddings_hflip1 = backbone(hflip1); embeddings_hflip2 = backbone(hflip2)
                  embeddings_rotate1 = backbone(rotate1); embeddings_rotate2 = backbone(rotate2)
                  # Check divice
                  # print(f'image1:{image1.device}, backbone:{backbone.device}, embeddings_image1:{embeddings_image1.device}')

                  embeddings1 = embeddings_image1+embeddings_hflip1+embeddings_rotate1
                  embeddings2 = embeddings_image2+embeddings_hflip2+embeddings_rotate2

              elif self.args.transform[0]:
                  image1 = img1[0].to(device); image2 = img2[0].to(device)
                  hflip1 = img1[1].to(device); hflip2 = img2[1].to(device)
                  embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2) 
                  embeddings_hflip1 = backbone(hflip1); embeddings_hflip2 = backbone(hflip2)

                  embeddings1 = embeddings_image1+embeddings_hflip1
                  embeddings2 = embeddings_image2+embeddings_hflip2

              elif self.args.transform[1]:
                  image1 = img1[0].to(device); image2 = img2[0].to(device)
                  rotate1 = img1[1].to(device); rotate2 = img2[1].to(device)

                  embeddings_image1 = backbone(image1); embeddings_image2 = backbone(image2)
                  embeddings_rotate1 = backbone(rotate1); embeddings_rotate2 = backbone(rotate2)
                  
                  embeddings1 = embeddings_image1+embeddings_rotate1
                  embeddings2 = embeddings_image2+embeddings_rotate2

              else:
                  image1 = img1.to(device); image2 = img2.to(device)
                  # image1 = img1; image2 = img2
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
      best_threshold, best_f1_score, precision, recall, acc = get_threshold(
                                    self.thresholds, dist_arr, issame_arr)

      self.best_f1_list.append(best_f1_score)
      self.best_thr_list.append(best_threshold)
      logging.info(f'F1 Score-Best: {best_f1_score:.3f}, Thr-Best:{best_threshold:.4f}')
      if best_f1_score > self.highest_f1:
          self.highest_f1 = best_f1_score
          self.highest_thr = best_threshold
          logging.info(f'F1 Score-Highest: {self.highest_f1:.3f}, Thr-Highest: {self.highest_thr:.4f}')
      
      if self.writer is not None:
         self.writer.line(Y=np.array([best_threshold]), X=np.array([epoch]), win='valid', name='best_threshold', 
                    update=None if epoch==0 else 'append', opts=dict(title='Metrics', showlegend=True))
         self.writer.line(Y=np.array([best_f1_score]), X=np.array([epoch]), win='valid', name='best_f1_score', 
                    update='append', opts=dict(title='Metrics', showlegend=True))
         self.writer.line(Y=np.array([precision]), X=np.array([epoch]), win='valid', name='precision', 
                    update='append', opts=dict(title='Metrics', showlegend=True))
         self.writer.line(Y=np.array([recall]), X=np.array([epoch]), win='valid', name='recall', 
                    update='append', opts=dict(title='Metrics', showlegend=True))
         self.writer.line(Y=np.array([acc]), X=np.array([epoch]), win='valid', name='acc', 
                    update='append', opts=dict(title='Metrics', showlegend=True))
         
    def __call__(self, backbone: torch.nn.Module, epoch):
        if self.rank == 0:
            backbone.eval()
            self.ver_test(backbone, epoch)
            backbone.train()