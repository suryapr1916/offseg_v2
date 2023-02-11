# get necessary imports
import os
import h5py
from torch.utils.data import Dataset

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
                label_file = file
                label_subdir = subdir.split('/')[-1]
                self.label_list.append(os.path.join(label_dir, label_subdir, label_file))

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