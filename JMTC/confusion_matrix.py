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


def general():
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}

    city_true = data["cities"]
    device_true = data["devices"]
    scene_true = data["y_true"]
    scene_pre = data["y_pre"]

    confmat = confusion_matrix(y_true=scene_true, y_pred=scene_pre)

    all_num = np.sum(confmat,axis=1).reshape(10,1)
    confmat = confmat/all_num
    confmat = np.round(confmat,2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xticks(range(10), labels=labels,rotation=45,ha="left")
    plt.yticks(range(10), labels=labels,rotation=45,ha="right")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("The confusion matrix of test set")
    # plt.show()
    plt.savefig(os.path.join(save_add,"general.png"),dpi=120,bbox_inches='tight')

def device():
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}

    city_true = data["cities"]
    device_true = data["devices"]
    scene_true = data["y_true"]
    scene_pre = data["y_pre"]
    device_name = {1:'b',2:'c'}
    for d in range(1,3):
        index = device_true == d
        confmat = confusion_matrix(y_true=scene_true[index], y_pred=scene_pre[index])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        plt.xticks(range(10), labels=labels,rotation=45,ha="left")
        plt.yticks(range(10), labels=labels,rotation=45,ha="right")

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title("The confusion matrix of device {}".format(device_name[d]))
        png_add = os.path.join(save_add,"device")
        if not os.path.isdir(png_add):
            os.makedirs(png_add)
        plt.savefig(os.path.join(png_add,"{}.png".format(device_name[d])),dpi=120,bbox_inches='tight')



def location():
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}

    city_true = data["cities"]
    device_true = data["devices"]
    scene_true = data["y_true"]
    scene_pre = data["y_pre"]

    for c in range(10):
        index = city_true == c
        confmat = confusion_matrix(y_true=scene_true[index], y_pred=scene_pre[index])
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        plt.xticks(range(10), labels=labels,rotation=45,ha="left")
        plt.yticks(range(10), labels=labels,rotation=45,ha="right")

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title("The confusion matrix of city {}".format(city2index[c]))
        png_add = os.path.join(save_add,"city")
        if not os.path.isdir(png_add):
            os.makedirs(png_add)
        plt.savefig(os.path.join(png_add,"{}.png".format(city2index[c])),dpi=120,bbox_inches='tight')


def device_location():
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}

    city_true = data["cities"]
    device_true = data["devices"]
    scene_true = data["y_true"]
    scene_pre = data["y_pre"]
    device_name = {1:'b',2:'c'}
    all_diff = []
    for c in range(10):
        diff = []
        for d in range(1,3):
            index_c = city_true == c
            scene_true_c = scene_true[index_c]
            scene_pre_c = scene_pre[index_c]
            device_true_c = device_true[index_c]
            index = device_true_c == d
            confmat = confusion_matrix(y_true=torch.cat([scene_true_c[index],torch.arange(10)]), y_pred=torch.cat([scene_pre_c[index],torch.arange(10)]))
            all_num = np.sum(confmat, axis=1).reshape(10, 1)
            confmat = confmat / all_num
            diff.append(confmat)
            confmat = np.round(confmat, 2)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3,vmin=0,vmax=1)
            for i in range(confmat.shape[0]):
                for j in range(confmat.shape[1]):
                    ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center',fontsize=14)

            # plt.xticks(range(10), labels=labels,rotation=45,ha="left")
            # plt.yticks(range(10), labels=labels,rotation=45,ha="right")
            plt.xticks(range(10), labels=range(10),fontsize=14)
            plt.yticks(range(10), labels=range(10),fontsize=14)



            plt.xlabel('Predicted Label',fontsize=16)
            plt.ylabel('True Label',fontsize=16)
            # plt.title("The confusion matrix of city {}".format(city2index[c]))
            png_add = os.path.join(save_add,"device_city",device_name[d])
            if not os.path.isdir(png_add):
                os.makedirs(png_add)
            plt.savefig(os.path.join(png_add,"{}_{}.png".format(device_name[d],city2index[c])),dpi=120,bbox_inches='tight')

        all_diff.append(np.sum(np.abs(diff[0]-diff[1]))/diff[0].shape[0])
    print("all_diff: {}".format(np.mean(all_diff)))
# general()
# device()
# location()
device_location()




