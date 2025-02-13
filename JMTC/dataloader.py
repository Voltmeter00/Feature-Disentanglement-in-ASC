from torch.utils.data import Dataset
import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision.models as models
import torchvision.transforms.functional as TF
import json
import csv
import pickle
import h5py
import librosa

class Load_Data(Dataset):
    def __init__(self,data_root="/home/share/tyz/DCASE2019/unzip/TAU-urban-acoustic-scenes-2019-development/mel_spec",json_root='../json/cross_cd_mel.json',type="train",test_city_index=[0],train_city_index=[1]):

        self.type = type
        f = open(json_root)
        #  [city2index,scene2index,index2data]   index2data -->ws [add,city,scene]
        self.train_data,self.test_data = json.load(f)

        self.data_root = data_root

        if type == "train":
            self.length = len(self.train_data)
        else:
            self.length = len(self.test_data)

        self.len_test = len(self.test_data)

    def __len__(self):

        return  self.length

    def __getitem__(self, idx):
        if self.type =="train":
            add,city,scene,device = self.train_data[idx]
            sample = np.load(os.path.join(self.data_root,add),allow_pickle=True)
            sample = librosa.power_to_db(sample)
            sample = torch.from_numpy(sample.T)

            test_idx = random.randint(0,self.len_test-1)
            add_t,city_t,scene_t,device_t = self.test_data[test_idx]
            sample_t = np.load(os.path.join(self.data_root,add_t),allow_pickle=True)
            sample_t = librosa.power_to_db(sample_t)
            sample_t = torch.from_numpy(sample_t.T)
            return sample, city,scene,device,sample_t, city_t,scene_t,device_t # torch.Size([32, 431, 256])
        else:
            add_t,city_t,scene_t,device_t= self.test_data[idx]
            sample_t = np.load(os.path.join(self.data_root,add_t),allow_pickle=True)
            sample_t = librosa.power_to_db(sample_t)
            sample_t = torch.from_numpy(sample_t.T)
            return sample_t, city_t,scene_t,device_t





if __name__ == '__main__':

    train_dataset = Load_Data()
    model = models.resnet18()
    trainloader_s = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for i, (x,y, class_id) in enumerate(trainloader_s):
        print(x.size())
        print(y.size())
        break
