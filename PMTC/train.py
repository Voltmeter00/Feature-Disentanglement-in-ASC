from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataloader
import os
import imp
import model
import math
import time
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score
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
        self.train_dataset = dataloader.Load_Data(data_root=config["data_dir"],json_root=config["json_root"],type="train")
        self.trainloader = DataLoader(self.train_dataset, batch_size=data_train_opt['batch_size']*len(device_ids),num_workers=8,shuffle=True,drop_last=True)
        self.valid_dataset = dataloader.Load_Data(data_root=config["data_dir"],json_root=config["json_root"],type="val")
        self.validloader = DataLoader(self.valid_dataset,batch_size=data_train_opt['batch_size']*len(device_ids),num_workers=8,shuffle=True)
        print("Trainloader: {}".format(len(self.trainloader)))
        print("Validloader: {}".format(len(self.validloader)))

        self.model = model.Net(dim = data_train_opt["dim"],city_class_num=10,scene_class_num=10,K=data_train_opt["K"])
        self.model = self.model.cuda(device=device_ids[0])

        self.LossFun()
    def LossFun(self):
        print("lossing...")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.encoder.parameters(), lr=data_train_opt['lr'])
        self.optimizer_1 = optim.Adam(self.model.scene.parameters(), lr=data_train_opt['lr'])

        # state = torch.load(
        #     os.path.join("/home/share/tyz/ASC_min_dataset_DCASE_2019/experiments/DA_none_city_cls/dim_1024_lr0.0001_batch64", 'best_all_others_{}.pth'.format(self.test_city_index)))
        # state = state["state_dict"]
        # model_dict = self.model.state_dict()
        # print(state.keys())
        # print(model_dict.keys())
        # pretrained_dict = {}
        # for k, v in state.items():
        #     k_list = k.split(".")
        #     k_list[0] = "encoder"
        #     k = ".".join(k_list)
        #     if k in model_dict:
        #         if v.size() == model_dict[k].size():
        #             pretrained_dict[k] = v
        # print(pretrained_dict.keys())
        # model_dict.update(pretrained_dict)
        # self.model.load_state_dict(model_dict)
    def TrainingData(self):
        self.model.train()
        log = []
        for epoch in range(data_train_opt['epoch']):
            if (epoch+1) % data_train_opt["decay_epoch"] == 0 :
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*data_train_opt["decay_rate"]
                for param_group in self.optimizer_1.param_groups:
                    param_group['lr'] = param_group['lr']*data_train_opt["decay_rate"]

            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top_c = AverageMeter('City@1', ':6.2f')
            top_s = AverageMeter('Scene@1', ':6.2f')
            top_d = AverageMeter('Device@1', ':6.2f')
            progress = ProgressMeter(
                len(self.trainloader),
                [batch_time, data_time, losses,top_c,top_s,top_d],
                prefix="Epoch: [{}]".format(epoch+1))

            # switch to train mode
            self.model.train()
            end = time.time()
            for i, (sample, city,scene,device,sample_t, city_t,scene_t,device_t) in enumerate(self.trainloader):
                # measure data loading time
                len_dataloader = len(self.trainloader)
                p = float(i + epoch * len_dataloader) / data_train_opt['epoch'] / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                data_time.update(time.time() - end)


                sample, city, scene = sample.cuda(device=device_ids[0]),city.cuda(device=device_ids[0]),scene.cuda(device=device_ids[0])

                sample_t, city_t, scene_t = sample_t.cuda(device=device_ids[0]), city_t.cuda(device=device_ids[0]), scene_t.cuda(
                    device=device_ids[0])
                device, device_t =device.cuda(device=device_ids[0]), device_t.cuda(device=device_ids[0])

                pre_device,pre_city,f_1,f_2,t_1,t_2= self.model(x=sample,x_t=sample_t,type="train",alpha=alpha)

                batch = sample.size(0)

                all_city_label = torch.cat([city,city_t])
                all_device_label = torch.cat([device,device_t])



                loss = (self.criterion(f_1, scene) + self.criterion(f_2, scene)) - data_train_opt["mcd"] * torch.mean(
                    torch.sum(torch.abs(t_1 - t_2), dim=-1))

                # acc1/acc5 are (K+1)-way contrast classifier accuracy
                # measure accuracy and record loss
                c_acc= accuracy(pre_city, all_city_label, topk=(1,))
                d_acc = accuracy(pre_device, all_device_label, topk=(1,))
                s_acc= accuracy(f_1+f_2, scene, topk=(1,))

                losses.update(loss.item(), sample.size(0))
                top_c.update(c_acc[0], all_city_label.size(0))
                top_s.update(s_acc[0], scene.size(0))
                top_d.update(d_acc[0],all_device_label.size(0))
                self.optimizer_1.zero_grad()
                loss.backward()
                self.optimizer_1.step()

                for n in range(1):
                    pre_device,pre_city, f_1, f_2, t_1, t_2 = self.model(x=sample,x_t=sample_t,type="train",alpha=alpha)

                    # joint_pre_device = pre_device.unsqueeze(2)
                    # joint_pre_city = pre_city.unsqueeze(1)
                    # pre_city = joint_pre_device.bmm(joint_pre_city).view(batch+int(batch/2), -1)

                    # all_city_label = all_city_label + 10 * (all_device_label)
                    # all_city_label = all_city_label.long()

                    loss_c = (self.criterion(pre_city[:batch], city) + 0.25*self.criterion(pre_city[batch:], city_t))/1.25
                    loss_d = (0.5*self.criterion(pre_device[:batch], device) + self.criterion(pre_device[batch:], device_t))/1.5

                    loss = data_train_opt["mcd"] * torch.mean(torch.sum(torch.abs(t_1 - t_2), dim=-1)) + (
                                self.criterion(f_1, scene) + self.criterion(f_2, scene)) + data_train_opt["grl"] * (loss_c + loss_d)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if (i+1) % data_train_opt["log_step"] == 0:
                    loss_avg = losses.avg
                    acc_avg = top_s.avg
                    # log.append([epoch, i + 1, loss.item(), s_acc[0], loss_avg, acc_avg])
                    progress.display(i+1)

            if (epoch+1) % data_train_opt["save_epoch"] == 0:
                a = 0
                acc,acc_b,acc_c = self.ValidingData(epoch+1)
                if acc > self.best:
                    self.best = acc
                    a = 1
                acc_add = data_train_opt["acc_log"]
                if os.path.exists(acc_add):
                    f = open(acc_add)
                    train_list = json.load(f)
                    train_list.append([epoch+1,acc,acc_b,acc_c])
                    f.close()
                    with open(acc_add, "w") as f:
                        json.dump(train_list, f)
                else:
                    with open(acc_add, "w") as f:
                        json.dump([[epoch+1,acc,acc_b,acc_c]], f)
                if a == 1:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'acc':acc
                    # }, filename=os.path.join(data_train_opt["feat_training_file"],'Epoch_{}_acc_{}_loss_{}.pth'.format(epoch+1,acc,losses.avg)))
                    # }, filename=os.path.join(data_train_opt["feat_training_file"],'checkpoint_{:04d}.pth'.format(epoch+1)))
                    }, filename=os.path.join(data_train_opt["feat_training_file"],'best_model.pth'))


                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'acc': acc
                }, filename=os.path.join(data_train_opt["feat_training_file"], '{}.pth'.format(epoch+1)))

    def save_checkpoint(self,state,filename):
        torch.save(state, filename)
    def ValidingData(self,epoch):

        self.model.eval()
        a = 0
        devices = []
        with torch.no_grad():
            y_pre = []
            y_true = []
            with tqdm(total=len(self.validloader), desc='Example', leave=True, ncols=100, unit='batch', unit_scale=True) as pbar:
                for i, (sample,city,scene,device) in enumerate(self.validloader):
                    sample, city, scene,device = sample.cuda(device=device_ids[0]),city.cuda(device=device_ids[0]), scene.cuda(device=device_ids[0]), device.cuda(device=device_ids[0])
                    predict = self.model(x=None,x_t=sample,type="valid",alpha=0)
                    _, pre = torch.max(predict,dim=1)
                    y_pre.append(pre.cpu())
                    y_true.append(scene.cpu())
                    devices.append(device.cpu())
                    pbar.update(1)

            y_pre = torch.cat(y_pre).cpu().detach().numpy()
            y_true = torch.cat(y_true).cpu().detach().numpy()
            devices = torch.cat(devices).cpu().detach().numpy()
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

            print("acc: {:.4f}   acc_b: {:.4f}   acc_c:{:.4f}".format(acc,acc_b,acc_c))
            with open(data_train_opt["txt"],"a") as f:
                f.write("========= {} =======\n".format(epoch))
                f.write("acc: {:.4f}   acc_b: {:.4f}   acc_c:{:.4f}".format(acc,acc_b,acc_c))
                f.write("\n")

        self.model.train()

        return acc,acc_b,acc_c



def main():

    ImgCla = ImageClassify()
    ImgCla.TrainingData()


if __name__ == '__main__':
    main()