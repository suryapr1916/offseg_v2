## OFFSEG v2

## Stuff to keep in mind

- Change batch_size and max_epochs in config.file

- May have to optimize the gpu performance

- There is no normalization performed on the kmaps (hp_masks) i.e., they are not normalized to [0,1] range but the images are normalized to [0,1] range

- The loss has not been mentioned in the config file and has to be changed in the model class

- Make sure to add the 'data' folder in the root directory, refer to paths in config file