import json
import os.path
import imp
import numpy as np

top_k=3
config = imp.load_source("config", "config/config.py").config
data_train_opt = config['data_train_opt']
root = data_train_opt["feat_training_file"]
print(root)
all_mean = []
file = "acc_log.json"
f = open(os.path.join(root,file))
data = json.load(f)
data = np.array(data)
def cal(index):
    mean_acc = np.mean(data[-5:,index])
    max_acc = np.max(data[-15:,index])
    return mean_acc,max_acc

mean_acc,max_acc = cal(1)
mean_acc_b,max_acc_b = cal(2)
mean_acc_c,max_acc_c = cal(3)

acc = data[-30:,[0,1]].tolist()
acc_b = data[-30:,[0,2]].tolist()
acc_c = data[-30:,[0,3]].tolist()
print("Acc: {}".format(acc))
print("Acc_b: {}".format(acc_b))
print("Acc_c: {}".format(acc_c))
print("Acc: mean {:.4f} max {:.4f}".format(mean_acc,max_acc))
print("Acc_b: mean {:.4f} max {:.4f}".format(mean_acc_b,max_acc_b))
print("Acc_c: mean {:.4f} max {:.4f}".format(mean_acc_c,max_acc_c))