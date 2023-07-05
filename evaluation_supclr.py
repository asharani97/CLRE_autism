from __future__ import print_function

import sys
import argparse
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from dataset.data_test import VideoDataset
from util import AverageMeter
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, #0.1,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, #0,
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
    parser.add_argument('--setup', type=str, default='',
                        help='train set')
    parser.add_argument('--test_set', type=str, default='',
                        help='test set')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.dataset == 'path':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt
    
def set_loader(opt):
    #train and val loader defined here
    if opt.dataset == 'path':
        train_data = VideoDataset(dataset=opt.setup, split='train')
        val_data = VideoDataset(dataset=opt.setup, split=opt.test_set)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader



def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
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

    return total_loss / total_num, correct / total_num *100


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    total_num ,total_loss, correct= 0.0, 0.0, 0
    
    with torch.no_grad():  
        for idx, (images, labels,filename) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            total_num += images.size(0)

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            total_loss += loss.item() * images.size(0)

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


def main():
    best_acc = 0
    test_arr = []
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    #save accuarcies for future use
    results ={'train_loss': [], 'train_acc@1':[],
               'val_loss': [], 'val_acc@1': []}
    # training routine
    for epoch in range(1, opt.epochs + 1):
        #adjust_learning_rate(opt, optimizer, epoch)
        # train for one epoch
        train_loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(acc)
        # eval for one epoch
        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        test_arr.append(val_acc)
        results['val_loss'].append(val_loss)
        results['val_acc@1'].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
        #save stats
        data_frame = pd.DataFrame(data=results, index=range(1, epoch+1))
        data_frame.to_csv('supclr_results/'+opt.setup+'_'+opt.test_set+'linear_stats.csv',index_label='epoch')

    print('best accuracy: {:.4f}'.format(best_acc))


if __name__ == '__main__':
    main()
