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

def kl(a,b):
    n = 0
    for i in range(10):
        n += a[i]*torch.log(a[i]/b[i])
    return n

def js(a,b):
    ab = (a+b)/2
    return 0.5*kl(a,ab)+0.5*kl(b,ab)

def M0():
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}

    city_true = data["cities"]
    device_true = data["devices"]
    scene_true = data["y_true"]
    city_pre = data["city_pre"]
    city_pre = torch.softmax(city_pre,dim=1)
    _, city_hard = torch.max(city_pre, dim=1)
    city_hard = generate_one_hot(city_hard,10)
    index_b = device_true == 1
    index_c = device_true == 2

    b = city_true[index_b]
    c = city_true[index_c]
    pre_b = city_pre[index_b]
    pre_c = city_pre[index_c]

    result = []
    for i in range(10):
        city_index_b = b == i
        city_index_c = c == i
        mean_b = torch.mean(pre_b[city_index_b], dim=0)
        mean_c = torch.mean(pre_c[city_index_c], dim=0)
        result.append(torch.sum(js(mean_b,mean_c)).item())
    print(np.mean(result))



# general()
# device()
# location()
# device_location()
M0()



