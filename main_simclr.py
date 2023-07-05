import os
import torch
import argparse
import pandas as pd
import torch.nn as nn
from util import train
import torch.optim as optim
from dataset.data_train import VideoDataset
from networks.resnet_big import Model
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=256, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=16, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--setup',default ='', type =str ,
         help='Name of train folder')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature= args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data prepare
    train_data = VideoDataset(dataset=args.setup,split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    # model setup and optimizer config
    model = Model(feature_dim)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    
    # training loop
    batch_loss=[]
    results = {'train_loss': []}
    save_name_pre = '{}_{}_{}_{}'.format(feature_dim, temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer,batch_size,epoch,epochs)
        results['train_loss'].append(train_loss)
        batch_loss.append(train_loss)
        # save statistics
        torch.save(model.state_dict(), 'results/'+args.setup+'/{}_model.pth'.format(save_name_pre))
   
        
