from torch.utils.data import random_split
from utils import HDF5Dataset
import torch
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
import wandb
from models.vae import CONVAE
from PIL import Image

from torch.nn import functional as F

def init_test(checkpoint_path, test_dataloader):
    
    model = CONVAE()
    model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    
    # evaluate on all entries in the test set and show some samples
    model.eval()
    device = 'cpu'
    for i, (x, y) in enumerate(test_dataloader):
        if i%100 == 0:
    
            x = x.to(device)
            y = y.to(device)
    
            y_pred = model(x)
            
            print("Loss: ", F.mse_loss(y_pred, y))
            
            print("Original image:")
            # use pillow to show the image
            print(x.view(-1, 550, 688, 3)[0].cpu().numpy().shape)
            Image.fromarray(x.view(550, 688, 3,-1)[0].cpu().numpy()).show()
            
            print("K Means image:")
            print(y_pred.view(-1, 550, 688, 3)[0].cpu().detach().numpy().shape)
            Image.fromarray(y_pred.view(550, 688, 3,-1)[0].cpu().numpy()).show()
            
            
            print("Reconstructed image:")
            print(y.view(-1, 550, 688, 3)[0].cpu().detach().numpy().shape)
            Image.fromarray(y.view(550, 688, 3,-1)[0].cpu().numpy()).show()
            
    return None