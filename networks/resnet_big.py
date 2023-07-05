"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models #needed to use model directly
from vmz import r2plus1d_18

class Flatten(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Unsqueeze(nn.Module):
    """A shape adaptation layer to patch certain networks."""
    def __init__(self):
        super(Unsqueeze, self).__init__()

    def forward(self, x):
        return x.unsqueeze(-1)

#kaiming_normal weight intialization
def random_weight_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None: 
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='r2plus1d', num_classes=2):
        super(LinearClassifier, self).__init__()
        feat_dim = 512
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        features1 = torch.flatten(features, start_dim=1)
        return self.fc(features1)
        
        
class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='r2plus1d', head='mlp',pretrained=False,p=0.3,proj_dim = 256,n_hidden = 512, encoder_dim = 512): 
        super(SupConResNet, self).__init__()
        self.encoder = r2plus1d_18(pretrained=pretrained, larger_last=False)
        if not pretrained:
            print("Randomly initializing models")
            random_weight_init(self.encoder)
        if head == 'linear':
            self.head = nn.Linear(encoder_dim, num_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(Flatten(),nn.Dropout(p=p),
                nn.Linear(encoder_dim, n_hidden, bias=False),
                Unsqueeze(),nn.BatchNorm1d(n_hidden),
                Flatten(),nn.ReLU(inplace=True),nn.Dropout(p=p),nn.Linear(n_hidden, proj_dim, bias=True))
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x).squeeze()
        feat = torch.flatten(feat, start_dim=-1)
        feat = F.normalize(self.head(feat), dim=-1)
        return feat

"""Binary Classifier"""


class BinClassifier(nn.Module):
    def __init__(self, name='r2plus1d',pretrained=False):
        super(BinClassifier, self).__init__()
        self.encoder = r2plus1d_18(pretrained=pretrained, larger_last=False)
        if not pretrained:
            #print("Randomly initializing models")
            random_weight_init(self.encoder)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        feat = self.encoder(x)
        feat_sig = self.sig(feat)
        return feat, feat_sig

"""SimCLR model"""
class Model(nn.Module):
    def __init__(self, feature_dim=256):
        super(Model, self).__init__()

        self.f = []
        for name, module in r2plus1d_18(pretrained=False, larger_last=False).named_children():
            if not isinstance(module, nn.Linear) :
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Net(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(Net, self).__init__()

        # encoder
        self.f = Model(256).f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = ckpt
        new_state_dict = {}
        for k, v in state_dict.items():
            n = k.replace("module.", "")
            k = n.replace("f.","")
            new_state_dict[k] = v
        state_dict = new_state_dict
        #mlp head weights deleted
        del_list = ["g.0.weight", "g.1.weight", "g.1.bias", "g.1.running_mean", "g.1.running_var", "g.1.num_batches_tracked", "g.3.weight", "g.3.bias"]
        for i in del_list:
            del state_dict[i]

        self.f.load_state_dict(state_dict)
        self.features = nn.Sequential(*list(self.f.children())[:1])
        for p in self.features.parameters():
          p.requires_grad = True
      

    def forward(self, x):
        x = self.f(x)
        feature1 = torch.flatten(x, start_dim=1)
        out = self.fc(feature1)
        return out 

