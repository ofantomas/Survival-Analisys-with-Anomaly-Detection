import torch
import torch.nn.functional as F
from torch.utils import data as dt
from torch import nn
from tqdm import tqdm

class UnetAE(nn.Module):
    def __init__(self, in_channels=1, num_features=30, num_layers=3, num_filters=64,
                 BatchNorm=True, Attention=False, multidimensional=False):
        super(UnetAE, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.Attention = Attention
        
        self.feature_config = [num_features] + [num_features // (2 ** (i + 1)) for i in range(num_layers)]
        self.out_padding = [num_features % 2 for num_features in self.feature_config[:-1:]]
        self.out_channels = [num_filters * (2 ** i) for i in range(self.num_layers + 1)]
        self.in_channels = [in_channels] + self.out_channels[:self.num_layers]
        
        self.down_path = nn.Sequential(*[ConvBlock(self.in_channels[i], self.out_channels[i], BatchNorm=BatchNorm)
                                         for i in range(num_layers + 1)])
        if Attention is False:
            self.up_path = nn.Sequential(*[UpConvBlock(self.out_channels[-i - 1],
                                                       self.in_channels[-i - 1],
                                                       out_paddding=self.out_padding[-i - 1],
                                                       BatchNorm=BatchNorm) for i in range(num_layers)])
        else:
            self.up_path = nn.Sequential(*[AttentionBlock(self.out_channels[-i - 1],
                                                          self.in_channels[-i - 1],
                                                          out_paddding=self.out_padding[-i - 1],
                                                          BatchNorm=BatchNorm,
                                                          multidimensional=multidimensional) for i in range(num_layers)])
        self.last = nn.Conv2d(self.out_channels[0], 1, kernel_size=1)

    def forward(self, x):
        bridge_list = []
        for i in range(self.num_layers + 1):
            x = self.down_path[i](x)
            if i != self.num_layers:
                bridge_list.append(x)
                x = F.max_pool2d(x, kernel_size=2)
        
        for i in range(self.num_layers):
            x = self.up_path[i](x, bridge_list[-i - 1])
        
        return self.last(x)


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, padding=1, BatchNorm=True):
        super(ConvBlock, self).__init__()
        if BatchNorm is True:
            self.layers_list = nn.Sequential(nn.Conv2d(in_features, out_features, padding=padding, kernel_size=3), 
                                             nn.LeakyReLU(), nn.BatchNorm2d(out_features),
                                             nn.Conv2d(out_features, out_features, padding=padding, kernel_size=3),
                                             nn.LeakyReLU(), nn.BatchNorm2d(out_features))
        else:
            self.layers_list = nn.Sequential(nn.Conv2d(in_features, out_features, padding=padding, kernel_size=3), nn.LeakyReLU(),
                                             nn.Conv2d(out_features, out_features, padding=padding, kernel_size=3), nn.LeakyReLU())
    def forward(self, x):
        return self.layers_list(x)

    
class UpConvBlock(nn.Module):
    def __init__(self, in_features, out_features, padding=1, inner_padding=0, out_paddding=0, BatchNorm=True):
        super(UpConvBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_features, out_features, kernel_size=2,
                                           stride=2, padding=inner_padding, output_padding=out_paddding)
        self.convblock = ConvBlock(in_features, out_features, padding, BatchNorm)
    
    def forward(self, x, bridge):
        out = self.upsample(x)
        return self.convblock(torch.cat([out, bridge], 1))


class AttentionBlock(nn.Module):
    def __init__(self, in_features, out_features, padding=1, inner_padding=0, out_paddding=0, BatchNorm=True, multidimensional=False):
        super(AttentionBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_features, out_features, kernel_size=2,
                                           stride=2, padding=inner_padding, output_padding=out_paddding)
        self.convblock = ConvBlock(in_features, out_features, padding, BatchNorm)
        self.AG = AttentionGate(F_g=out_features, F_l=out_features, F_int=out_features // 2, 
                                BatchNorm=BatchNorm, multidimensional=multidimensional)

    def forward(self, x, bridge):
        out = self.upsample(x)
        bridge = self.AG(bridge, out)
        return self.convblock(torch.cat([out, bridge], 1))
    

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, BatchNorm=True, multidimensional=False):
        super(AttentionGate, self).__init__()
        
        self.W_g = [nn.Conv2d(F_g, F_int, kernel_size=1)]
        if BatchNorm is True:
            self.W_g += [nn.BatchNorm2d(F_int)]
        self.W_g = nn.Sequential(*self.W_g)
        
        self.W_l = [nn.Conv2d(F_l, F_int, kernel_size=1)]
        if BatchNorm is True:
            self.W_l += [nn.BatchNorm2d(F_int)]
        self.W_l = nn.Sequential(*self.W_l)
        
        if multidimensional is True:
            self.psi = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(2 * F_int, F_l, kernel_size=1), nn.Sigmoid())
        else:
            self.psi = nn.Sequential(nn.LeakyReLU(), nn.Conv2d(2 * F_int, 1, kernel_size=1), nn.Sigmoid())
            
    def forward(self, g, x):
        int_x = self.W_l(x)
        int_g = self.W_g(g)
        #try to cat features instead!
        return x * self.psi(torch.cat([int_x, int_g], 1))
        
def train_model(model, data, device, batch_size=64, num_epochs=5, learning_rate=1e-3, mu=0.1, 
                test_func=None, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau):
    gd = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = scheduler(gd, 'min', patience=5)
    criterion = nn.MSELoss(reduction='mean')

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        dataloader = dt.DataLoader(data, batch_size=batch_size, shuffle=True, timeout=0.5)
    
        with tqdm(enumerate(dataloader), total=len(dataloader)) as iterator:
            for batch_num, batch in iterator:
                gd.zero_grad()
                x, target = batch
                out = model(x.to(device))
                loss = criterion(out, target.to(device).unsqueeze(1)) + mu * criterion(out, out.permute(0, 1, 3, 2))
                
                loss.backward()
                gd.step()

                train_losses.append(float(loss))            
                iterator.set_description('Train loss: %.5f' % train_losses[-1])
        
        val_loss = test_func(model.eval())
        scheduler.step(val_loss)
        test_losses.append(val_loss)
        model.train()
        
    return train_losses, test_losses 

def get_mse_scores(model, data, device, batch_size=64, reduction=None):
    dataloader = dt.DataLoader(data, batch_size=batch_size)
    mse_scores = []
    
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=len(dataloader)) as iterator:
            for batch_num, batch in iterator:
                x, target = batch
                outputs = model(x.to(device))
                #print(outputs.shape, target.to(device).unsqueeze(1).shape, ((outputs - target.to(device).unsqueeze(1)) ** 2).shape, torch.mean((outputs - target.to(device).unsqueeze(1)) ** 2, (2, 3)).shape)
                mse_scores.append(torch.mean((outputs - target.to(device).unsqueeze(1)) ** 2, (2, 3)).cpu().reshape(-1, 1))

    res = torch.cat(mse_scores, 0)
    
    if reduction is not None:
        if reduction == 'sum':
            return torch.sum(res)
        if reduction == 'max':
            return torch.max(res)
        if reduction == 'mean':
            return torch.mean(res)
        else:
            raise(AttributeError("Unknown reduction type"))
    else:
        return res
        