# Deep Image Reconstruction from Human Brain Activity
## 541 MSc Advanced Computing Individual Project, Imperial College London
### Created by: Bin Xuan Kong
### Supervised by: Yi-ke Guo, Pan Wang

Data and demo code for deep image reconstruction from human brain activity. The method is based on [Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. PLOS Computational Biology](http://dx.doi.org/10.1371/journal.pcbi.1006633) with a few modifications to tailor to our own dataset. Their code is available [here](https://github.com/KamitaniLab/DeepImageReconstruction).

This project is created in partial fulfillment of the requirements for the MSc degree in Advanced Computing of Imperial College London.

## Requirements

- Python 3
- Numpy
- PyTorch
- Pickle
- Pillow (PIL)
- [slir](https://github.com/KamitaniLab/slir)
- [BigGAN](https://github.com/huggingface/pytorch-pretrained-BigGAN)

## Usage

### Setup

1. Download data files from figshare (see [data/README.md](data/README.md)).
2. Install Python requirements
```
pip install -r requirements.txt
```

### Brain activity decoding

### Image reconstructions from decoded features
