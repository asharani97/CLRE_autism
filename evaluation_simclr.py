import torch
import argparse
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from data_test import VideoDataset
from networks.resnet_big import Net
from torch.utils.data import DataLoader


def train(net,train_loader,optimizer):
    """one epoch training"""
    net.train()
    correct ,total_num, total_loss, data_bar = 0, 0.0, 0.0, tqdm(train_loader)
    for data, target, filename in data_bar:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        out = net(data)
        loss = loss_criterion(out, target)
        total_num += data.size(0)
        # compute loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = out.argmax(dim=1)
        correct += torch.eq(pred, target).sum().float().item()
        data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                     .format('Train', epoch, epochs, total_loss / total_num,correct / total_num * 100))

    return total_loss / total_num, correct / total_num *100


def val(net,test_loader):
    """one epoch training"""
    net.eval()
    correct ,total_num, total_loss, data_bar, output = 0, 0.0, 0.0, tqdm(test_loader),[]
    for data, target, filename in test_loader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        out = net(data)
        loss = loss_criterion(out, target)
        total_num += data.size(0)

        # compute loss
        total_loss += loss.item() * data.size(0)
        pred = out.argmax(dim=1)
        correct += torch.eq(pred, target).sum().float().item()
        predictions = pred.data.cpu().tolist()
        target = target.data.cpu().tolist()
        output=[predictions,target]
        data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}%'
                                     .format('Test', epoch, epochs, total_loss / total_num,correct / total_num * 100))

    return total_loss / total_num, correct / total_num *100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='/DATA/rani.1/feb/weights/new/simclr/modal2/256_0.07_16_200_lr_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=200, help='Number of sweeps over the dataset to train')
    parser.add_argument('--setup', type=str, default='', 
                help='Specify the setup/dataset')
    parser.add_argument('--test_set', type=str,default='',
                         help='split to be tested')
    args = parser.parse_args()
    model_path, batch_size, epochs = args.model_path, args.batch_size, args.epochs
    
    train_data = VideoDataset(dataset=args.setup, split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_data = VideoDataset(dataset=args.setup, split=args.test_set)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = Net(num_class=2, pretrained_path=model_path)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model=torch.nn.DataParallel(model)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)#,momentum=0.9)
    loss_criterion = nn.CrossEntropyLoss().cuda()
    results = {'train_loss': [], 'train_acc@1': [],
               'test_loss': [], 'test_acc@1': []}

    best_acc = 0.0
    test_arr=[]
    for epoch in range(1, epochs + 1):
        train_loss, train_acc_1 = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        results['train_acc@1'].append(train_acc_1)
        test_loss, test_acc_1 = val(model, val_loader)
        test_arr.append(test_acc_1)
        if best_acc<test_acc_1:
            best_acc=test_acc_1
            torch.save(model.state_dict(), 'results/'+'best_model.pth')
        results['test_loss'].append(test_loss)
        results['test_acc@1'].append(test_acc_1)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('simclr_results/'+args.setup+'_'+args.test_set+'linear_statistics.csv', index_label='epoch')
    print("Best ACC @1",best_acc)
