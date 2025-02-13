import json
import os.path
import imp
import numpy as np

last_n = 30
epoch = 30
config = imp.load_source("config", "config/config.py").config
data_train_opt = config['data_train_opt']
root = data_train_opt["feat_training_file"]
all_data = []
for i in range(5):
    file = "acc_log_{}.json".format(i)
    f = open(os.path.join(root,file))
    data = json.load(f)
    data = data[-epoch:]
    arr = np.array(data)[:,1]
    all_data.append(data)

all_data = np.array(all_data)[:,-last_n:,1]
print(all_data)
all_max = np.max(all_data,axis=1)
print(np.round(all_max*100,1))
# all_std = np.std(all_data,axis=1)
# print(all_std)



print("{} {}".format(round(np.mean(all_max)*100,1),(round(np.min(all_max)*100,1),round(np.max(all_max)*100,1))))

#
# print(round(np.mean(all_std)*100,2))