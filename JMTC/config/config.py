import os

config = {}
data_train_opt = {}
config["data_dir"] = "/share/database/TAU Urban Acoustic Scenes 2019 Mobile Development dataset/mel_spec"
config["json_root"] = '/home/tyz/ASC_cross_city_device_k_fold/json/cross_cd_mel_fold_{}.json'
config["device_ids"] = [0]
data_train_opt["dim"] = 1024
data_train_opt["K"] = 256
data_train_opt['batch_size'] = 32
data_train_opt['epoch'] = 30
data_train_opt['split'] = 'train'
data_train_opt['lr'] = 0.0001
data_train_opt["decay_epoch"] = 10
data_train_opt["decay_rate"] = 0.1
data_train_opt["save_epoch"] = 1
data_train_opt["log_step"] = 10
data_train_opt["continue_model"] = ""
data_train_opt["mcd"] = 0.1
data_train_opt["grl"] = 0.1
feat_training_file = '/share/tyz/ASC_experiments/k_fold/Cross_dc_JMTC_2/dim_{}_lr{}_batch{}_mcd_{}_grl_{}'.format(data_train_opt["dim"],data_train_opt['lr'],data_train_opt['batch_size'],data_train_opt["mcd"],data_train_opt["grl"])
final_model_file = os.path.join(feat_training_file,"Final_model.pth")
data_train_opt["acc_log"] = os.path.join(feat_training_file,"acc_log_{}.json")
data_train_opt["training_log"] = os.path.join(feat_training_file,"training_log.npy")
data_train_opt["txt"] = os.path.join(feat_training_file,"acc.txt")
if not os.path.exists(feat_training_file):
    os.makedirs(feat_training_file)
data_train_opt["best"] = os.path.join(feat_training_file,"acc_best.txt")


data_train_opt["feat_training_file"] = feat_training_file
data_train_opt["final_model_file"] = final_model_file
config["data_train_opt"] = data_train_opt


