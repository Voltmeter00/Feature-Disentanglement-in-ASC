import json
import os.path
import imp
import numpy as np
import torch
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

save_add = "png"
if not os.path.isdir(save_add):
    os.makedirs(save_add)

f = open('/home/tyz/ASC/json/DG_city_scene_mel.json')
#  [city2index,scene2index,index2data]   index2data -->ws [add,city,scene]
device2data, city2index, scene2label = json.load(f)
print(scene2label)
labels = list(scene2label.keys())
labels.sort()
print(labels)
print(city2index)
city2index = list(city2index.keys())
print(city2index)

config = imp.load_source("config", "config/config.py").config
data_train_opt = config['data_train_opt']
root = data_train_opt["feat_training_file"]
print(root)
all_mean = []

def generate_one_hot(index,class_num):
    index = index.unsqueeze(1)
    a = torch.zeros(index.size(0), class_num).scatter_(1, index, 1)
    return a



def M1():
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}

    city_true = data["cities"]
    device_true = data["devices"]
    scene_true = data["y_true"]
    scene_pre = data["all_predict"]
    index_b = device_true == 1
    index_c = device_true == 2
    def xxx(index):
        distribution = []
        index_city = city_true[index]
        index_scene = scene_pre[index]
        for i in range(10):
            c_index = index_city == i
            sum_all = torch.mean(index_scene[c_index],dim=0)
            distribution.append(sum_all.unsqueeze(0))
        return torch.cat(distribution,dim=0)

    print(torch.mean(torch.sum(torch.abs(xxx(index_b) - xxx(index_c)),dim=1)))




# general()
# device()
# location()
# device_location()
M1()



