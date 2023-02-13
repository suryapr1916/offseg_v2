from torch.utils.data import random_split
from utils import HDF5Dataset
import torch
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
import wandb

def get_transforms():
    data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (255, 255., 255.))
    ])

    label_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (255., 255., 225.))
    ])
    return [data_transform, label_transform]

def create_dataset(config):
    both_transforms = get_transforms()
    train_dataset = HDF5Dataset(config.INPUT_DIR, config.OUTPUT_DIR, transform=both_transforms)
    return train_dataset

def get_dataloaders(config):
    train_dataset = create_dataset(config)
    total_count = train_dataset.__len__()
    train, test = random_split(train_dataset,
                 [int(total_count*config.train_split), total_count - int(total_count*config.train_split)])
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=config.batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def finish_wandb():
    wandb.finish()
    
def init_wandb(config):
    wandb.init()
    return WandbLogger(project=config.project_name)

def init_train(config, logger = None):
    if not logger:
        logger = init_wandb(config)
    trainer = Trainer(logger=logger,
                callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience = 5)],
                max_epochs=config.max_epochs,
                accelerator=config.accelerator,
                default_root_dir=config.CKPT_DIR,)
    return trainer