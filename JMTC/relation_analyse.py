import json
import os.path
import imp
import numpy as np
import torch
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
top_k=3
config = imp.load_source("config", "config/config.py").config
data_train_opt = config['data_train_opt']
root = data_train_opt["feat_training_file"]
print(root)
all_mean = []
f = open('/home/tyz/ASC/json/DG_city_scene_mel.json')
#  [city2index,scene2index,index2data]   index2data -->ws [add,city,scene]
device2data, city2index, scene2label = json.load(f)
print(scene2label)
labels = list(scene2label.keys())
labels.sort()
print(labels)
print(city2index)
city2index = list(city2index.keys())
city2index.sort()
print(city2index)
save_add = "png"
if not os.path.isdir(save_add):
    os.makedirs(save_add)
def generate_one_hot(index,class_num):
    index = index.unsqueeze(1)
    a = torch.zeros(index.size(0), class_num).scatter_(1, index, 1)
    return a

def train_true():
    print("real train")
    file = 'train_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    city_pre = data["cities"]
    city_pre = generate_one_hot(city_pre,10)
    device_pre = data["devices"]
    device_pre = generate_one_hot(device_pre,3)
    city_pre = torch.softmax(city_pre,dim=1)
    device_pre = torch.softmax(device_pre,dim=1)

    city_pre_mean = torch.mean(city_pre,dim=0).unsqueeze(0)
    device_pre_mean = torch.mean(device_pre,dim=0).unsqueeze(0)

    city_pre_diff = city_pre - city_pre_mean
    device_pre_diff = device_pre - device_pre_mean

    device_pre_diff = device_pre_diff.unsqueeze(2)
    city_pre_diff = city_pre_diff.unsqueeze(1)

    cov_matrix = torch.bmm(device_pre_diff,city_pre_diff)

    cov_matrix = torch.mean(cov_matrix,dim=0)
    print(cov_matrix)
    print(torch.mean(torch.abs(cov_matrix)))

def test_true():
    print("real test")
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    city_pre = data["cities"]
    city_pre = generate_one_hot(city_pre,10)
    device_pre = data["devices"]
    device_pre = generate_one_hot(device_pre,3)
    city_pre = torch.softmax(city_pre,dim=1)
    device_pre = torch.softmax(device_pre,dim=1)

    city_pre_mean = torch.mean(city_pre,dim=0).unsqueeze(0)
    device_pre_mean = torch.mean(device_pre,dim=0).unsqueeze(0)

    city_pre_diff = city_pre - city_pre_mean
    device_pre_diff = device_pre - device_pre_mean

    device_pre_diff = device_pre_diff.unsqueeze(2)
    city_pre_diff = city_pre_diff.unsqueeze(1)

    cov_matrix = torch.bmm(device_pre_diff,city_pre_diff)

    cov_matrix = torch.mean(cov_matrix,dim=0)
    print(cov_matrix)
    print(torch.mean(torch.abs(cov_matrix)))


def test():
    print("test")
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    city_pre = data["city_pre"]
    device_pre = data["device_pre"]

    city_pre = torch.softmax(city_pre,dim=1)
    device_pre = torch.softmax(device_pre,dim=1)

    city_pre_mean = torch.mean(city_pre,dim=0).unsqueeze(0)
    device_pre_mean = torch.mean(device_pre,dim=0).unsqueeze(0)

    city_pre_diff = city_pre - city_pre_mean
    device_pre_diff = device_pre - device_pre_mean

    device_pre_diff = device_pre_diff.unsqueeze(2)
    city_pre_diff = city_pre_diff.unsqueeze(1)

    cov_matrix = torch.bmm(device_pre_diff,city_pre_diff)
    cov_matrix = torch.mean(cov_matrix,dim=0)
    cov_matrix = cov_matrix*(10**4)
    mean_value = torch.mean(torch.abs(cov_matrix)).item()

    cov_matrix = cov_matrix.numpy()
    cov_matrix = np.round(cov_matrix, 2)
    mean_value = np.round(mean_value, 2)
    print(cov_matrix)
    print(mean_value)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.matshow(cov_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            ax.text(x=j, y=i, s=cov_matrix[i, j], va='center', ha='center')

    plt.xticks(range(10), labels=city2index, rotation=45, ha="left")
    plt.yticks(range(3), labels=["a","b","c"])

    plt.xlabel('City')
    plt.ylabel('Device')
    plt.title("Mean of absolute value: {}     Scale: 10^-4".format(mean_value))
    png_add = os.path.join(save_add, "covariance")
    if not os.path.isdir(png_add):
        os.makedirs(png_add)
    plt.savefig(os.path.join(png_add, "covariance.png"), dpi=120, bbox_inches='tight')



def train():
    print("++++++++++++train+++++++++++++++")
    file = 'train_result.pth'
    data = torch.load(os.path.join(root, file))
    # {"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    city_pre = data["city_pre"]
    device_pre = data["device_pre"]

    city_pre = torch.softmax(city_pre, dim=1)
    device_pre = torch.softmax(device_pre, dim=1)

    city_pre_mean = torch.mean(city_pre, dim=0).unsqueeze(0)
    device_pre_mean = torch.mean(device_pre, dim=0).unsqueeze(0)

    city_pre_diff = city_pre - city_pre_mean
    device_pre_diff = device_pre - device_pre_mean

    device_pre_diff = device_pre_diff.unsqueeze(2)
    city_pre_diff = city_pre_diff.unsqueeze(1)

    cov_matrix = torch.bmm(device_pre_diff, city_pre_diff)
    cov_matrix = torch.mean(cov_matrix,dim=0)
    mean_value = torch.mean(torch.abs(cov_matrix)).item()
    print(cov_matrix)
    print(mean_value)
    # fig, ax = plt.subplots(figsize=(10, 3))
    # ax.matshow(cov_matrix, cmap=plt.cm.Blues, alpha=0.3)
    # for i in range(cov_matrix.shape[0]):
    #     for j in range(cov_matrix.shape[1]):
    #         ax.text(x=j, y=i, s=cov_matrix[i, j], va='center', ha='center')
    #
    # plt.xticks(range(10), labels=city2index, rotation=45, ha="left")
    # plt.yticks(range(10), labels=["a","b","c"])
    #
    # plt.xlabel('City')
    # plt.ylabel('Device')
    # plt.title("Average absolute value: {}".format(mean_value))
    # png_add = os.path.join(save_add, "covariance")
    # if not os.path.isdir(png_add):
    #     os.makedirs(png_add)
    # plt.savefig(os.path.join(png_add, "covariance.png", dpi=120, bbox_inches='tight'))


def test_confidence():
    print("test")
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    city_pre = data["city_pre"]
    device_pre = data["device_pre"]

    city_pre = torch.softmax(city_pre, dim=1)
    device_pre = torch.softmax(device_pre, dim=1)

    entropy_city = torch.mean(torch.sum(-city_pre*torch.log(city_pre),dim=1))
    entropy_device = torch.mean(torch.sum(-device_pre*torch.log(device_pre),dim=1))

    print("Entropy City: {}".format(entropy_city))
    print("Entropy Device: {}".format(entropy_device))
    city_true = data["cities"]
    device_true = data["devices"]

    _, city_pre = torch.max(city_pre, dim=1)
    _, device_pre = torch.max(device_pre, dim=1)

    city_acc = accuracy_score(city_true, city_pre)
    device_acc = accuracy_score(device_true, device_pre)
    print("city_acc: {}".format(city_acc))
    print("device_acc: {}".format(device_acc))


def train_confidence():
    print("train")
    file = 'train_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    city_pre = data["city_pre"]
    device_pre = data["device_pre"]

    city_pre = torch.softmax(city_pre, dim=1)
    device_pre = torch.softmax(device_pre, dim=1)

    entropy_city = torch.mean(torch.sum(-city_pre*torch.log(city_pre+0.0000001),dim=1))
    entropy_device = torch.mean(torch.sum(-device_pre*torch.log(device_pre),dim=1))

    print("Entropy City: {}".format(entropy_city))
    print("Entropy Device: {}".format(entropy_device))
    city_true = data["cities"]
    device_true = data["devices"]

    _, city_pre = torch.max(city_pre, dim=1)
    _, device_pre = torch.max(device_pre, dim=1)

    city_acc = accuracy_score(city_true, city_pre)
    device_acc = accuracy_score(device_true, device_pre)
    print("city_acc: {}".format(city_acc))
    print("device_acc: {}".format(device_acc))


# train_true()
# test_true()
train()
test()
# test_confidence()
# train_confidence()










