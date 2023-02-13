# get necessary imports
import os
import h5py
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class HDF5Dataset(Dataset):

  def __init__(self, root_dir, label_dir, transform = None):
    self.missing = []
    self.root_dir = root_dir
    self.label_dir = label_dir
    self.file_list = []
    self.label_list = []
    self.transform = transform
    # subdir does not return just the directory
    for subdir, _, files in os.walk(self.root_dir):
        for file in files:
            if file.endswith(".hdf5"):
                self.file_list.append(os.path.join(subdir, file))
                # this \\ instead of / alone took half my fucking time
                self.label_list.append(os.path.join(self.label_dir, subdir.split('\\')[-1], file))

  def __len__(self):
    for data_file, label_file in zip(self.file_list, self.label_list):
      if not os.path.exists(data_file):
        self.missing.append('image,' + data_file)
      if not os.path.exists(label_file):
        self.missing.append('label,' + label_file)
    print('Missing files!' if len(self.missing) > 0 else 'No missing files found.')
    return len(self.file_list)

  def __getitem__(self, idx):
    with h5py.File(self.file_list[idx], "r") as f:
      data = f[list(f.keys())[0]][()]
    with h5py.File(self.label_list[idx], "r") as f:
      label = f[list(f.keys())[0]][()]

    if self.transform:
      data = self.transform[0](data).float()
    label = self.transform[1](label).float()
    
    return data, label

def convert_ext(my_string,to_ext):
  return os.path.splitext(my_string)[0] + '.' + to_ext

# a function that takes a tensor (range 0-1) and converts it to an image (range 0-255)
# and saves the image to the specified path using PIL or matplotlib
def tensor_to_image(tensor, path):
  # convert to numpy
  image = tensor.numpy()
  # transpose to (channels, height, width)
  image = np.transpose(image, (1, 2, 0))
  # scale to 0-255
  image = image * 255
  # convert to uint8
  image = image.astype(np.uint8)
  # save image
  plt.imsave(path, image)
