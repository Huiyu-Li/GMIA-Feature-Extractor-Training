## GMIA-Feature Extractor Training &mdash; Official PyTorch implementation

![](./docs/feature_extractor_training.png)

**Generative Medical Image Anonymization Based on Latent Code Projection and Optimization**<br>
Huiyu Li, Nicholas Ayache, Hervé Delingette<br>
<!-- ToDo<br> -->
[**Data Exfiltration and Anonymization of Medical Images based on Generative Models** (Chapter3-4)](https://inria.hal.science/tel-04875160)<br>
Huiyu Li<br>

## Requirements
* 64-bit Python 3.9 pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.

## Getting started

## Preparing Datasets
**MIMIC-CXR-JPG**:<br>
Step 1: Download the [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.1.0/).

Step 2: Dataset Pre-processing accoring to the [repository](https://github.com/Huiyu-Li/GMIA-Dataset-Pre-processing/tree/main).

## Training the Networks

### Identity network training
Train the idenity network on the original dataset.<br>
```.bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 --nnodes=1 ./Model_Pretrain/identity_classification.py
```

Train the idenity network on the anonymized dataset.<br>
```.bash
OMP_NUM_THREADS=8 torchrun --nproc_per_node=2 --nnodes=1 ./Model_Pretrain/identityA_classification.py
```

### Utility network training
Train the utility network on the original dataset.<br>
```.bash
python3 ./Model_Pretrain/utility_classification_CheXclusion_uDense.py
```

Train the utility network on the anonymized dataset.<br>
```.bash
python3 ./Model_Pretrain/utilityA_classification_CheXclusion_uDense.py
```

## Evaluating the Networks
### Identity network evaluation
Evaluate the identity network on the original dataset.<br>
```.bash
python3 ./Evaluate/identity_eval.py
```

Evaluate the Inner linability risk of the anonymized dataset.<br>
```.bash
python3 ./Evaluate/identityI_eval.py
```

Evaluate the Outer linability risk of the anonymized dataset.<br>
```.bash
python3 ./Evaluate/identityO_eval.py
```

### Utility network evaluation
Evaluate the utility network on the original dataset.<br>
```.bash
python3 ./Evaluate/utility_eval_CheXclusion_uDense.py
```

Evaluate the utility network on the anonymized dataset.<br>
```.bash
python3 ./Evaluate/utilityA_eval_CheXclusion_uDense.py
```

### Privacy metrics
Calculate the privacy metrics in the identity feature space.<br>
```.bash
python3 ./Privacy_Metrics/main_identity.py
```

Calculate the privacy metrics in the latent space.<br>
```.bash
python3 ./Privacy_Metrics/main_latent.py
```

References:
1. [Distributed Arcface Training in Pytorch](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) 
2. [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
3. [libauc.models.densenet](https://docs.libauc.org/_modules/libauc/models/densenet.html#DenseNet)
4. [Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification](https://arxiv.org/abs/2012.03173)
5. [Avatar](https://github.com/octopize/avatar-paper)

## Citation

<!-- ToDo<br> -->
```
@article{li2025generative,
  title={Generative medical image anonymization based on latent code projection and optimization},
  author={Li, Huiyu and Ayache, Nicholas and Delingette, Herv{\'e}},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI 2025)},
  year={2025}
}
@phdthesis{li2024data,
  title={Data exfiltration and anonymization of medical images based on generative models},
  author={Li, Huiyu},
  year={2024},
  school={Universit{\'e} C{\^o}te d'Azur}
}
```