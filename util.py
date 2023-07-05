"""SupContrast: Supervised Contrastive Learning
ADAPTED from https://github.com/HobbitLong/SupContrast """
from __future__ import print_function
import math
import numpy as np
import torch
import torch.optim as optim
from losses import nt_xent_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def train(net, data_loader, train_optimizer,batch_size,epoch,epochs):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_1, x_2 in train_bar:
        x_1, x_2 = x_1.cuda(non_blocking=True), x_2.cuda(non_blocking=True)
        h_1, z_1 = net(x_1)
        h_2, z_2 = net(x_2)
        loss=nt_xent_loss(z_1,z_2,0.07)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num