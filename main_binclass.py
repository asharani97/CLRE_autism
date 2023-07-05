
import os
import sys
import argparse
import torch
from util import AverageMeter
from torchvision import datasets
import torch.backends.cudnn as cudnn
from util import set_optimizer, save_model
from networks.resnet_big import BinClassifier
from dataset data_test import VideoDataset

seed=2021
torch.manual_seed(seed)
torch.backends.cudnn.deterministic= True
torch.backends.cudnn.benchmark = False


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')

    # model dataset
    parser.add_argument('--model', type=str, default='r2plus1d')
    parser.add_argument('--dataset', type=str, default='path',choices=['path'] ,help='dataset')
   
    # other setting
  
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--setup',type=str, default='',
                        help='setup on which model to be trained')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_path = './save/{}_models/{}'.format(opt.dataset,opt.setup)
    opt.tb_path = './save/{}_tensorboard/{}'.format(opt.dataset,opt.setup)

    

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)


    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt): 
    if opt.dataset == 'path':
        train_dataset = VideoDataset(dataset=opt.setup,split='train')
    else:
        raise ValueError(opt.dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    return train_loader


def set_model(opt):
    model = BinClassifier(name=opt.model)
    criterion = torch.nn.BCELoss()
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    
    losses = AverageMeter()

    for idx, (video, labels,) in enumerate(train_loader):   
        if torch.cuda.is_available():
            video = video.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True).float()
        bsz = labels.shape[0]

        #compute loss
        features = model(video)
        loss = criterion(features, labels)
        
        # update metric
        losses.update(loss.item(), bsz)

        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # print info
        if (idx + 1) % opt.print_freq == 0:
             print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses))
             sys.stdout.flush()

    return losses.avg
    


def main():
    best_acc=0
    test_arr =[]
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)
  
    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
    
        #adjust_learning_rate(opt, optimizer, epoch)
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
      

if __name__ == '__main__':
    main()
