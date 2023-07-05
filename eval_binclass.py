from __future__ import print_function

import sys
import argparse
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import accuracy_score
from data_test import VideoDataset
from util import AverageMeter
from util import set_optimizer
from networks.resnet_big import BinClassifier
from statistics import mean

seed=2021
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic= True
torch.backends.cudnn.benchmark = False


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='r2plus1d')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['path'], help='dataset')

    # other setting
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--test_set', type=str, default='',
                         help='name of the setup on which test to be performed')
    parser.add_argument('--setup', type=str, default='',
                         help = 'name of the setup using')
    parser.add_argument('--file',type=str,default ='',
                        help = 'name of file to write prediction')
    parser.add_argument('--name', type=str,default='',
                        help='name storing file')
    opt = parser.parse_args()

    # set the path according to the environment

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.dataset == 'path':
        opt.n_cls = 1
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_loader(opt):

    if opt.dataset == 'path':
        val_data = VideoDataset(dataset=opt.setup, split=opt.test_set)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)

    return val_loader #, val_loader



def set_model(opt):
    model = BinClassifier(name=opt.model)
    #for param in model.parameters():
      #  param.requires_grad = False
    #criterion = torch.nn.BCELoss()

    #classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                print(k)
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
    #classifier = classifier.cuda()
    #criterion = criterion.cuda()
    cudnn.benchmark = True

    model.load_state_dict(state_dict,strict=True)
    #model = model.cuda()
    model = model.cuda()
    #classifier = classifier.cuda()

    return model


'''def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    classifier.train()

    losses = AverageMeter()
    correct = 0
    total_num = 0.0
    total_loss =0.0
    for idx, (images, labels) in enumerate(train_loader):
      
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        total_num += images.size(0)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        pred = output.argmax(dim=1)
        correct += torch.eq(pred, labels).sum().float().item()
        
        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
    

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss:.3f}\t'
                  'Acc@1: {acc:.2f}'.format(epoch, idx + 1, len(train_loader),loss=(total_loss / total_num), acc=((correct / total_num) *100)))
            sys.stdout.flush()

    return total_loss / total_num, correct / total_num *100'''


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    total_num ,total_loss, correct= 0.0, 0.0, 0


    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            total_num += images.size(0)

            # forward
            output = model(images)


            pred = output.argmax(dim=1)
            correct += torch.eq(pred, labels).sum().float().item()


            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss:.4f} \t'
                      'Acc@1: {acc:.3f}'.format(
                       idx,len(val_loader),
                       loss=(total_loss / total_num), acc=((correct / total_num)*100)))

    print(' * Acc@1 {:.3f}'.format((correct / total_num)*100))
    return total_loss / total_num, ((correct / total_num)*100)

def test(model, test_loader,opt):
    #model in eval mode skips Dropout etc
    model.eval()
    y_true = []
    y_pred = []
    prt=[]
    name1=[]
    # set the requires_grad flag to false as we are in the test mode
    with torch.no_grad():
        for i in test_loader:

            #LOAD THE DATA IN A BATCH
            data,target,file_name = i
            # moving the tensors to the configured device
            data = i[0].float().cuda()
            target = i[1].cuda()
            #name1.append(file_name)
            # the model on the data
            
            output = model(data.float())
            #PREDICTIONS
            pred = np.round(output.cpu())
            target = target.float()
            p1=pred.reshape(-1).tolist()
            t1 = target.tolist()
            y_true.extend(target.tolist()) 
            y_pred.extend(pred.reshape(-1).tolist())
            prt = [p1, t1]
            with open(opt.setup+'_'+opt.test_set+'_.txt','a') as c:
                c.write(f'{prt}\n')
    
    test_acc=accuracy_score(y_true,y_pred)
    
    print("Accuracy on test set in this epoch is " , test_acc)
    return test_acc



def main():
    best_acc = 0
    test_arr = []
    opt = parse_option()

    # build data loader
    val_loader = set_loader(opt)

    # build model and criterion
    model = set_model(opt)

    # build optimizer
    #optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
    
        #adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        #loss, acc = train(train_loader, model, classifier, criterion,optimizer, epoch, opt)
        
        # eval for one epoch
        #loss, val_acc = validate(val_loader, model, criterion, opt)
        print("----------------Epoch-----------------",epoch)
        val_acc = test_acc = test(model, val_loader,opt)
        test_arr.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
    print('best accuracy: {:.4f}'.format(best_acc))
    print('Average accuracy: {:.4f};'.format(mean(test_arr)))


if __name__ == '__main__':
    main()
