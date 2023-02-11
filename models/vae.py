from torch import nn, exp, randn_like
from torch.nn import functional as F
from torch.optim import Adam
import pytorch_lightning as pl

class CONVAE(pl.LightningModule):

	def __init__(self, image_channels=3, init_channels=8, kernel_size=3, intermediate_dim=384, latent_dim=32):
		super().__init__()
		# poolers and up-samplers

		self.flatten = nn.Flatten()
		self.unflatten = nn.Unflatten(1, (init_channels*8, 2, 3))
		
		# convolutional encoder
		self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
				stride=4, padding=0)

		self.conv2 = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size,
				stride=4, padding=0)

		self.conv3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size,
				stride=4, padding=0)

		self.conv4 = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size,
				stride=4, padding=0)

		# representation layer
		self.hidden1 = nn.Linear(intermediate_dim, init_channels*8)
		self.fc_mu = nn.Linear(init_channels*8, latent_dim)
		self.fc_log_var = nn.Linear(init_channels*8, latent_dim)
		self.fc_back = nn.Linear(latent_dim, init_channels*8)
		self.hidden2 = nn.Linear(init_channels*8, intermediate_dim)
		
		self.deconv1 = nn.ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size,
				stride = 4, output_padding=(1,0))
		
		self.deconv2 = nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size,
				stride = 4, output_padding=(3,0))
		
		self.deconv3 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=init_channels, kernel_size=kernel_size,
				stride = 4, output_padding=(2,1))
		
		self.deconv4 = nn.ConvTranspose2d(in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size,
				stride = 4, output_padding=(3,1))
	
	def reparameterize(self, mu, log_var):
		"""
		:param mu: mean from the encoder's latent space
		:param log_var: log variance from the encoder's latent space
		"""
		std = exp(0.5*log_var) # standard deviation
		eps = randn_like(std) # `randn_like` as we need the same size
		sample = mu + (eps * std) # sampling
		return sample

	def forward(self, x0):
		# encoder
		x1 = F.relu(self.conv1(x0))
		x2 = F.relu(self.conv2(x1))
		x3 = F.relu(self.conv3(x2))
		x4 = F.relu(self.conv4(x3))
		x = self.flatten(x4)
		x = F.relu(self.hidden1(x))

		# representation layer
		mu = self.fc_mu(x)
		log_var = self.fc_log_var(x)
		z = self.reparameterize(mu, log_var)
		
		# decoder
		x = F.relu(self.fc_back(z))
		x = F.relu(self.hidden2(x))
		x = self.unflatten(x)
		x = F.relu(self.deconv1(x)) + x3
		x = F.relu(self.deconv2(x))
		x = F.relu(self.deconv3(x)) + x1
		x = self.deconv4(x)
		return x

	def configure_optimizers(self):
		optimizer = Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		x, y = train_batch
		r = self.forward(x)
		loss = F.mse_loss(r, y)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		x, y = val_batch
		r = self.forward(x)    
		loss = F.mse_loss(r, y)
		self.log('val_loss', loss)
		return loss