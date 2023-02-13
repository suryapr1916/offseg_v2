from torch.utils.data import random_split
from models.vae import CONVAE
import numpy as np

from torch.nn import functional as F

def init_test(checkpoint_path, test_dataloader):
    
    model = CONVAE()
    model.load_from_checkpoint(checkpoint_path=checkpoint_path)
    
    # evaluate on all entries in the test set and show some samples
    model.eval()
    device = 'cpu'
    
    for i, (x, y) in enumerate(test_dataloader):

        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        
        print("Loss: ", F.mse_loss(y_pred, y))
        
        np.save('og.npy',x.view(-1, 550, 688, 3)[0].cpu().numpy())
        np.save('rec.npy',y_pred.view(-1, 550, 688, 3)[0].cpu().detach().numpy())
        np.save('kmi.npy',y.view(-1, 550, 688, 3)[0].cpu().numpy())
        
        break
    
    return None