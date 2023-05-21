import numpy as np
from torch.utils.data import TensorDataset
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Dataset class

class DataSet():
    def __init__(self, dataset = 'mnist', dataset_path = './data', batch_size = 1000, seed = 2, validation_split = 0.1, conf = None):
        if conf is None:
            self.dataset = dataset
            self.dataset_path = dataset_path
            self.batch_size = batch_size
            self.seed = seed
            self.validation_split = validation_split

        else:
            self.dataset = conf['dataconf']['dataset']
            self.dataset_path = conf['dataconf']['dataset_path']
            self.batch_size = conf['dataconf']['batch_size']
            self.seed = conf['dataconf']['seed']
            self.validation_split = conf['dataconf']['validation_split']
        

    def get_binary_dataloaders(self, classes = [0, 1]):
        if self.dataset == 'mnist':

            mnist = fetch_openml('mnist_784',version=1, parser='auto', data_home = './data/sklearn', cache=True)
            imgs_raw = mnist.data
            targets = mnist.target

            imgs = np.array(imgs_raw)/255
            targets = np.array(targets, dtype = np.float32)

            cls0 = targets==classes[0]
            cls1 = targets==classes[1]

            cls_y0 = targets[cls0]
            cls_x0 = imgs[cls0]

            cls_y1 = targets[cls1]
            cls_x1 = imgs[cls1]


            imgs_c = np.concatenate((cls_x0,cls_x1), axis=0)
            targets_c = np.concatenate((cls_y0,cls_y1), axis=0)
            targets_c = np.expand_dims(targets_c, axis = 1)

            np.random.seed(self.seed)
            p = np.random.permutation(imgs_c.shape[0])
            imgs_c = torch.tensor(imgs_c[p], dtype = torch.float32)
            targets_c = torch.tensor(targets_c[p]).type(torch.float32)

            # print(imgs_c.shape, imgs_c.min(), imgs_c.max())
            # print(targets_c.shape, targets_c.min(), targets_c.max())

            # Spliting dataset

            X_train, X_val, y_train, y_val = train_test_split(imgs_c, targets_c, test_size = self.validation_split, random_state=self.seed)

            train_dataset = TensorDataset(X_train, y_train) 
            val_dataset = TensorDataset(X_val, y_val)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)
            validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size)

            return train_loader, validation_loader 
