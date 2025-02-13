from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataloader
import os
import imp
import model_test as model
import math
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import json
import random

config = imp.load_source("config", "config/config.py").config
device_ids = config["device_ids"]
data_train_opt = config['data_train_opt']
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("======================================")
print("Device: {}".format(device_ids))


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k]
            correct_k = torch.sum(correct_k).float()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ImageClassify(object):
    def __init__(self):
        self.name_list = []
        self.save = data_train_opt["final_model_file"]
        self.training_log = data_train_opt["training_log"]
        self.loss = 9999
        self.best = 0
        self.d = None
        print("========================= New Training ===========================")
        self.train_dataset = dataloader.Load_Data(data_root=config["data_dir"], json_root=config["json_root"],
                                                  type="train")
        self.trainloader = DataLoader(self.train_dataset, batch_size=data_train_opt['batch_size'] * len(device_ids),
                                      num_workers=8, shuffle=True, drop_last=True)
        self.valid_dataset = dataloader.Load_Data(data_root=config["data_dir"], json_root=config["json_root"],
                                                  type="val")
        self.validloader = DataLoader(self.valid_dataset, batch_size=data_train_opt['batch_size'] * len(device_ids),
                                      num_workers=8, shuffle=True)
        print("Trainloader: {}".format(len(self.trainloader)))
        print("Validloader: {}".format(len(self.validloader)))

        self.model = model.Net(dim=data_train_opt["dim"], city_class_num=10, scene_class_num=10, K=data_train_opt["K"])
        self.model = self.model.cuda(device=device_ids[0])

        self.LossFun()

    def LossFun(self):
        print("lossing...")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.encoder.parameters(), lr=data_train_opt['lr'])
        self.optimizer_1 = optim.Adam(self.model.scene.parameters(), lr=data_train_opt['lr'])

        state = torch.load(os.path.join(data_train_opt["feat_training_file"], 'best_model.pth'))
        state = state["state_dict"]
        self.model.load_state_dict(state,strict=True)



    def Testset(self):

        self.model.eval()
        a = 0
        devices = []
        cities = []
        with torch.no_grad():
            y_pre = []
            city_pre = []
            device_pre = []
            all_feature = []
            y_true = []
            all_predict = []
            with tqdm(total=len(self.validloader), desc='Example', leave=True, ncols=100, unit='batch',
                      unit_scale=True) as pbar:
                for i, (sample, city, scene, device) in enumerate(self.validloader):
                    sample, city, scene, device = sample.cuda(device=device_ids[0]), city.cuda(
                        device=device_ids[0]), scene.cuda(device=device_ids[0]), device.cuda(device=device_ids[0])
                    pre_device,pre_city,predict,feature = self.model(x=None, x_t=sample, type="valid", alpha=0)
                    _, pre = torch.max(predict, dim=1)
                    all_predict.append(predict.cpu())
                    all_feature.append(feature.cpu())
                    y_pre.append(pre.cpu())
                    y_true.append(scene.cpu())
                    city_pre.append(pre_city.cpu())
                    device_pre.append(pre_device.cpu())
                    devices.append(device.cpu())
                    cities.append(city.cpu())
                    pbar.update(1)
            all_feature = torch.cat(all_feature).cpu().detach()
            city_pre = torch.cat(city_pre).cpu().detach()
            device_pre = torch.cat(device_pre).cpu().detach()
            y_pre = torch.cat(y_pre).cpu().detach()
            y_true = torch.cat(y_true).cpu().detach()
            devices = torch.cat(devices).cpu().detach()
            cities = torch.cat(cities).cpu().detach()
            all_predict = torch.cat(all_predict).cpu().detach()
            torch.save({"all_predict":all_predict,"all_feature":all_feature,"city_pre":city_pre,"device_pre":device_pre,"y_pre":y_pre,"y_true":y_true,"devices":devices, "cities": cities},
                       os.path.join(data_train_opt["feat_training_file"], 'test_result.pth'))

            b_index = devices == 1
            c_index = devices == 2

            # f = open(config["json_root"])

            # device2data,scene2label = json.load(f)
            # target_names = [i for i in range(10)]
            # for k in list(scene2label.keys()):
            #     target_names[scene2label[k]] = k
            # report = classification_report(y_true, y_pre, target_names= target_names, digits=4)
            acc = accuracy_score(y_true, y_pre)
            acc_b = accuracy_score(y_true[b_index], y_pre[b_index])
            acc_c = accuracy_score(y_true[c_index], y_pre[c_index])

            print("acc: {:.4f}   acc_b: {:.4f}   acc_c:{:.4f}".format(acc, acc_b, acc_c))

    def Trainset(self):

        self.model.eval()
        a = 0
        devices = []
        cities = []
        with torch.no_grad():
            y_pre = []
            city_pre = []
            device_pre = []
            y_true = []
            with tqdm(total=len(self.trainloader), desc='Example', leave=True, ncols=100, unit='batch',
                      unit_scale=True) as pbar:
                for i, (sample, city,scene,device,sample_t, city_t,scene_t,device_t) in enumerate(self.trainloader):
                    sample, city, scene, device = sample.cuda(device=device_ids[0]), city.cuda(
                        device=device_ids[0]), scene.cuda(device=device_ids[0]), device.cuda(device=device_ids[0])
                    pre_device, pre_city, predict = self.model(x=None, x_t=sample, type="valid", alpha=0)
                    _, pre = torch.max(predict, dim=1)
                    y_pre.append(pre.cpu())
                    y_true.append(scene.cpu())
                    city_pre.append(pre_city.cpu())
                    device_pre.append(pre_device.cpu())
                    devices.append(device.cpu())
                    cities.append(city.cpu())
                    pbar.update(1)

            city_pre = torch.cat(city_pre).cpu().detach()
            device_pre = torch.cat(device_pre).cpu().detach()
            y_pre = torch.cat(y_pre).cpu().detach()
            y_true = torch.cat(y_true).cpu().detach()
            devices = torch.cat(devices).cpu().detach()
            cities = torch.cat(cities).cpu().detach()

            torch.save(
                {"city_pre": city_pre, "device_pre": device_pre, "y_pre": y_pre, "y_true": y_true, "devices": devices,
                 "cities": cities},
                os.path.join(data_train_opt["feat_training_file"], 'train_result.pth'))

            b_index = devices == 1
            c_index = devices == 2

            # f = open(config["json_root"])

            # device2data,scene2label = json.load(f)
            # target_names = [i for i in range(10)]
            # for k in list(scene2label.keys()):
            #     target_names[scene2label[k]] = k
            # report = classification_report(y_true, y_pre, target_names= target_names, digits=4)
            acc = accuracy_score(y_true, y_pre)
            acc_b = accuracy_score(y_true[b_index], y_pre[b_index])
            acc_c = accuracy_score(y_true[c_index], y_pre[c_index])

            print("acc: {:.4f}   acc_b: {:.4f}   acc_c:{:.4f}".format(acc, acc_b, acc_c))



def main():
    ImgCla = ImageClassify()
    ImgCla.Testset()
    # ImgCla.Trainset()


if __name__ == '__main__':
    main()