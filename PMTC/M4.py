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
def M4():
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}

    city_true = data["cities"]
    device_true = data["devices"]
    scene_true = data["y_true"]
    scene_pre = data["all_predict"]
    index_b = device_true == 1
    index_c = device_true == 2

    city_b = city_true[index_b]
    city_c = city_true[index_c]
    scene_pre_b = scene_pre[index_b]
    scene_pre_c = scene_pre[index_c]
    scene_true_b = scene_true[index_b]
    scene_true_c = scene_true[index_c]
    result = 0
    for n in range(10):
        city_b_n_index = city_b == n
        city_c_n_index = city_c == n
        scene_pre_b_n = scene_pre_b[city_b_n_index]
        scene_pre_c_n = scene_pre_c[city_c_n_index]
        scene_true_b_n = scene_true_b[city_b_n_index]
        scene_true_c_n = scene_true_c[city_c_n_index]
        for m in range(1,10):
            index_b_m_n = scene_true_b_n == m
            index_c_m_n = scene_true_c_n == m

            # print(torch.sum(index_b_m_n))
            # print(torch.sum(index_c_m_n))
            a = torch.mean(scene_pre_b_n[index_b_m_n],dim=0)
            b = torch.mean(scene_pre_c_n[index_c_m_n],dim=0)
            result += js(a,b).item()
            # print(result)
    print(result/90)





M4()



