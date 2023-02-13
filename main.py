import hydra
from models.vae import CONVAE
from omegaconf import DictConfig, OmegaConf
from utils.train import *
from utils.test import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path="configs", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # get data loaders after preprocessing
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    # initialize wandb and trainer
    wandb_logger = init_wandb(cfg)
    trainer = init_train(cfg, logger=wandb_logger)

    # initalize model
    model = CONVAE()

    # run training
    trainer.fit(model, train_dataloader, val_dataloader)

@hydra.main(config_path="configs", config_name="config")
def test(cfg: DictConfig):
    # get data loaders after preprocessing
    _, val_dataloader = get_dataloaders(cfg)
    
    init_test(r'C:\Users\ragav\Code\offseg_v2\epoch=8-step=819.ckpt', val_dataloader)    

if __name__ == '__main__':
    # run()

    test()
