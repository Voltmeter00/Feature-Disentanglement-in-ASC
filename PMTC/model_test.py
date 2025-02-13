#-*- coding = utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pickle
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
import pickle
from torch.utils.data import DataLoader
import numpy as np
import cnn14
from functions import ReverseLayerF
class Net(nn.Module):
    def __init__(self,dim=1024,city_class_num=10,scene_class_num=10,K=256):
        super(Net, self).__init__()

        self.encoder = encoder(dim=1024,city_class_num=10,device_class_num=3)
        self.scene = scene(dim=1024,scene_class_num=10)


    def forward(self,x,x_t,type,alpha):

        if type == "train":
            batch = x.size(0)
            pre_device,pre_city,all_x_feature = self.encoder(x, x_t,type, alpha)
            feature_s = all_x_feature[:batch]
            feature_t = all_x_feature[batch:]
            t_1,t_2 = self.scene(feature_t)
            t_1 = F.softmax(t_1,dim=-1)
            t_2 = F.softmax(t_2,dim=-1)
            f_1,f_2 = self.scene(feature_s)


            return pre_device[int(batch/2):],pre_city[int(batch/2):],f_1,f_2,t_1,t_2
        else:
            pre_device,pre_city,x_t = self.encoder(x, x_t,type, alpha)
            result_1,result_2=self.scene(x_t)
            result_1 = F.softmax(result_1,dim=-1)
            result_2 = F.softmax(result_2,dim=-1)

            return pre_device,pre_city,(result_1+result_2)/2,x_t



class encoder(nn.Module):
    def __init__(self,dim=1024,city_class_num=10,device_class_num=3):
        super(encoder, self).__init__()
        self.audio_net = cnn14.AudioNet(dim=dim)
        self.city = nn.Sequential(
            nn.Linear(dim,int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),city_class_num),
        )
        self.device = nn.Sequential(
            nn.Linear(dim,int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),device_class_num),
        )
        self.shared_layer = nn.Sequential(
            nn.Linear(dim,dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
        )
    def forward(self,x, x_t,type, alpha):
        if type == "train":
            batch = x.size(0)
            all_x = torch.cat([x,x_t],dim=0)
            all_x_feature = self.audio_net(all_x)

            reverse_feature = ReverseLayerF.apply(all_x_feature, alpha)
            pre_city = self.city(self.shared_layer(all_x_feature))
            pre_device = self.device(self.shared_layer(reverse_feature))
            return pre_device,pre_city,all_x_feature
        else:
            all_x_feature = self.audio_net(x_t)
            pre_city = self.city(all_x_feature)
            pre_device = self.device(all_x_feature)
            return pre_device,pre_city,all_x_feature

class scene(nn.Module):
    def __init__(self,dim=1024,scene_class_num=10):
        super(scene, self).__init__()
        self.scene_1 = nn.Sequential(
            nn.Linear(dim,int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),scene_class_num),
        )
        self.scene_2 = nn.Sequential(
            nn.Linear(dim,int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5),scene_class_num),
        )

    def forward(self, x):
        return self.scene_1(x),self.scene_2(x)