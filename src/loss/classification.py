import torch
import torch.nn as nn

class BCEwithClassWeights(nn.Module):
    # 2023_CheXFusion: Effective Fusion of Multi-View Features using Transformers for Long-Tailed Chest X-Ray Classification_ICCV
    def __init__(self, class_instance_nums, n_instance):
        super(BCEwithClassWeights, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32) # [1, n_class]
        p = class_instance_nums / n_instance 
        self.pos_weights = torch.exp(1-p) # [1, n_class]
        self.neg_weights = torch.exp(p) # [1, n_class]

    def forward(self, pred, label): # Shape: [N, N_class]
        # https://www.cse.sc.edu/~songwang/document/cvpr21d.pdf (equation 4)
        self.pos_neg_weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()
        loss = nn.functional.binary_cross_entropy_with_logits(pred, label, weight=self.pos_neg_weight)
        return loss