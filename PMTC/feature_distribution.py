import json
import os.path
import imp
import numpy as np
import torch
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
import json
import os.path
import torch
import numpy as np
import torch
from sklearn.decomposition import PCA
import pandas as pd
import copy
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
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
for i in range(len(city2index)):
    city2index[i] = city2index[i][:2]
print(city2index)
save_add = "png"
if not os.path.isdir(save_add):
    os.makedirs(save_add)


def gmm_js(gmm_p, gmm_q, n_samples=10 ** 5):
    X= gmm_p.sample(n_samples)

    log_p_X = gmm_p.score_samples(X)

    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2

def generate_name(train_set):
    name = ""
    for x in train_set:
        name += str(x)
    return name

def choose_data(all_feature,all_city,index,city_label,all_scene,scene):
    feature = all_feature[index]
    all_scene = all_scene[index]
    city = all_city[index]
    new_index = city == city_label


    all_scene = all_scene[new_index]
    feature = feature[new_index]
    second_index= all_scene == scene
    return feature[second_index]



def js_city_in_bc(dim = 32):
    # calculate the js divergence B_ij = js(city_i,city_j) in device B

    bandwith = 1
    # reduced dim after pca
    print("real test")
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    all_city = data["cities"]
    devices = data["devices"]
    all_feature = data["all_feature"]
    all_scene = data["y_true"]
    random_times = 5
    index_list = []
    last = 0
    num_scene = 10
    pca = PCA(n_components=dim)
    all_feature = pca.fit_transform(all_feature)
    data_per = np.sum(pca.explained_variance_ratio_)
    print(data_per)
    b_index = devices == 1
    c_index = devices == 2


    js_matrix = np.zeros((10,10))

    for scene in range(2,3):
        for city_1 in range(10):
            feature_1 = choose_data(all_feature, all_city, b_index, city_1,all_scene,scene)
            print(feature_1.shape)
            if feature_1.shape[0] == 0:
                print("city : {} scene: {}".format(city_1,scene))
                continue
            gmm_1 = KernelDensity(bandwidth=bandwith).fit(feature_1)
            for city_2 in range(city_1+1,10):
                feature_2 = choose_data(all_feature, all_city, b_index, city_2,all_scene,scene)
                gmm_2 = KernelDensity(bandwidth=bandwith).fit(feature_2)
                js_value = gmm_js(gmm_1,gmm_2)
                js_matrix[city_1,city_2] = js_value

        for city_1 in range(10):
            feature_1 = choose_data(all_feature, all_city, c_index, city_1,all_scene,scene)
            if feature_1.shape[0] == 0:
                print("city : {} scene: {}".format(city_1,scene))
                continue
            gmm_1 = KernelDensity(bandwidth=bandwith).fit(feature_1)
            for city_2 in range(city_1 + 1, 10):
                feature_2 = choose_data(all_feature, all_city, c_index, city_2,all_scene,scene)
                gmm_2 = KernelDensity(bandwidth=bandwith).fit(
                    feature_2)
                js_value = gmm_js(gmm_1, gmm_2)
                js_matrix[city_2, city_1] = js_value

    print(np.round(js_matrix,4))
    with open(os.path.join(js_matrix,"js_city_bc_dim{}.json".format(dim)), "w") as f:
        json.dump(js_matrix, f)

def mean_city_in_bc(dim = 32):
    # calculate the js divergence B_ij = js(city_i,city_j) in device B

    bandwith = 1
    # reduced dim after pca
    print("real test")
    file = 'test_result.pth'
    data = torch.load(os.path.join(root,file))
    #{"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities}
    all_city = data["cities"]
    devices = data["devices"]
    all_feature = data["all_feature"]
    all_scene = data["y_true"]
    random_times = 5
    index_list = []
    last = 0
    num_scene = 10
    pca = PCA(n_components=dim)
    all_feature = pca.fit_transform(all_feature)
    data_per = np.sum(pca.explained_variance_ratio_)
    print(data_per)
    b_index = devices == 1
    c_index = devices == 2

    def same_city_vector():
        all_vector_matrix = np.zeros((10, 10, 9, dim))
        all_distance_matrix = np.zeros((10, 10))
        for city_1 in range(10):
            vector_matrix = np.zeros((9, dim))
            for scene in range(1, 10):
                feature_1 = choose_data(all_feature, all_city, b_index, city_1, all_scene, scene)
                if feature_1.shape[0] == 0:
                    print("city : {} scene: {}".format(city_1, scene))
                    continue
                mean_1 = np.median(feature_1, axis=0)
                feature_2 = choose_data(all_feature, all_city, c_index, city_1, all_scene, scene)
                if feature_2.shape[0] == 0:
                    print("city : {} scene: {}".format(city_1, scene))
                    continue
                mean_2 = np.median(feature_2, axis=0)
                vector = mean_1 - mean_2
                # distance = np.sqrt(np.sum(vector*vector))
                # direction = vector/np.sqrt(np.sum(vector*vector))
                vector_matrix[scene - 1] = vector
            distance_mean = np.mean(np.sqrt(np.sum(vector_matrix ** 2, axis=1)))
            all_vector_matrix[city_1, city_1] = vector_matrix
            all_distance_matrix[city_1,city_1]=distance_mean

        return all_vector_matrix, all_distance_matrix

    def generate_vector_matrix_distance(device_index):
        all_vector_matrix = np.zeros((10,10,9,dim))
        all_distance_matrix = np.zeros((10,10))
        for city_1 in range(10):
            for city_2 in range(city_1+1,10):
                vector_matrix = np.zeros((9,dim))
                for scene in range(1, 10):
                    feature_1 = choose_data(all_feature, all_city, device_index, city_1, all_scene,scene)
                    if feature_1.shape[0] == 0:
                        print("city : {} scene: {}".format(city_1, scene))
                        continue
                    mean_1 = np.median(feature_1, axis=0)
                    feature_2 = choose_data(all_feature, all_city, device_index, city_2,all_scene,scene)
                    if feature_2.shape[0] == 0:
                        print("city : {} scene: {}".format(city_1, scene))
                        continue
                    mean_2 = np.median(feature_2, axis=0)
                    vector = mean_1 - mean_2
                    # distance = np.sqrt(np.sum(vector*vector))
                    # direction = vector/np.sqrt(np.sum(vector*vector))
                    vector_matrix[scene-1] = vector
                distance_mean = np.mean(np.sqrt(np.sum(vector_matrix**2,axis=1)))
                all_vector_matrix[city_1,city_2] = vector_matrix
                all_vector_matrix[city_2,city_1] = vector_matrix
                all_distance_matrix[city_1,city_2] = distance_mean
                all_distance_matrix[city_2,city_1] = distance_mean

        return all_vector_matrix,all_distance_matrix
    b_vector_matrix,b_distance_matrix = generate_vector_matrix_distance(b_index)
    c_vector_matrix,c_distance_matrix = generate_vector_matrix_distance(c_index)
    same_vector_matrix,same_distance_matrix = same_city_vector()

    print(np.round(b_distance_matrix,1))
    print(np.round(c_distance_matrix,1))
    print(np.round(same_distance_matrix,1))
    cos_similarity = np.ones((10,10))

    diff_distance = np.sqrt(np.sum(b_vector_matrix ** 2, axis=3)) * np.sqrt(np.sum(c_vector_matrix ** 2, axis=3))

    multiple_matrix =np.sum(b_vector_matrix*c_vector_matrix,axis=3)
    for city_1 in range(10):
        for city_2 in range(city_1+1,10):

            m_distance = diff_distance[city_1,city_2]
            sum_distance = np.sum(diff_distance[city_1,city_2])
            distance_weight = m_distance/sum_distance
            data = multiple_matrix[city_1,city_2]


            value = np.sum(distance_weight*data/m_distance)

            cos_similarity[city_1,city_2] = value
            cos_similarity[city_2,city_1] = value
    print(cos_similarity)
    def draw_cos_similarity():
        fontsize = 16
        confmat = np.round(cos_similarity,2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3,vmin=0.5, vmax=1)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center',fontsize=12)

        # plt.xticks(range(10), labels=city2index, rotation=45, ha="left")
        # plt.yticks(range(10), labels=city2index, rotation=45, ha="right")
        plt.xticks(range(10), labels=city2index, fontsize=fontsize-2)
        plt.yticks(range(10), labels=city2index, fontsize=fontsize-2)

        plt.xlabel('City',fontsize=fontsize-2)
        plt.ylabel('City',fontsize=fontsize-2)
        # plt.title("The cosine similarity of city bias between device b and c",fontsize=fontsize)
        png_add = os.path.join(save_add, "feature_bias")
        if not os.path.isdir(png_add):
            os.makedirs(png_add)
        plt.savefig(os.path.join(png_add, "cosine_similarity.png"), dpi=120, bbox_inches='tight')

    def draw_b_distance():
        confmat = np.round(b_distance_matrix,2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        plt.xticks(range(10), labels=city2index, rotation=45, ha="left")
        plt.yticks(range(10), labels=city2index, rotation=45, ha="right")

        plt.xlabel('City')
        plt.ylabel('City')
        plt.title("The mean bias of scenes among cities in device b")
        png_add = os.path.join(save_add, "feature_bias")
        if not os.path.isdir(png_add):
            os.makedirs(png_add)
        plt.savefig(os.path.join(png_add, "b_distance.png"), dpi=120, bbox_inches='tight')

    def draw_c_distance():
        confmat = np.round(c_distance_matrix,2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        plt.xticks(range(10), labels=city2index, rotation=45, ha="left")
        plt.yticks(range(10), labels=city2index, rotation=45, ha="right")

        plt.xlabel('City')
        plt.ylabel('City')
        plt.title("The mean bias of scenes among cities in device c")
        png_add = os.path.join(save_add, "feature_bias")
        if not os.path.isdir(png_add):
            os.makedirs(png_add)
        plt.savefig(os.path.join(png_add, "c_distance.png"), dpi=120, bbox_inches='tight')

    def same_city():
        confmat = np.round(same_distance_matrix,2)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

        plt.xticks(range(10), labels=city2index, rotation=45, ha="left")
        plt.yticks(range(10), labels=city2index, rotation=45, ha="right")

        plt.xlabel('City')
        plt.ylabel('City')
        plt.title("The mean bias of scenes between device b and c in same cities")
        png_add = os.path.join(save_add, "feature_bias")
        if not os.path.isdir(png_add):
            os.makedirs(png_add)
        plt.savefig(os.path.join(png_add, "same.png"), dpi=120, bbox_inches='tight')

    draw_cos_similarity()
    draw_b_distance()
    draw_c_distance()
    same_city()









# js_city_in_bc()
mean_city_in_bc()









