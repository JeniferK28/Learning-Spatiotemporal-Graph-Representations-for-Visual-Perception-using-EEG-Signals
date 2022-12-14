import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

class Net(torch.nn.Module):
    def __init__(self, num_node_feat, num_classes, time_points,time_kernel, channels ,pool_kernel, hidden_channels, features, device):
        super(Net, self).__init__()
        self.channels = channels
        self.timepoints = time_points
        self.device = device
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, time_kernel), padding=0),
                                      nn.BatchNorm2d(16),
                                      nn.ELU(True))
        self.conv2 = nn.Sequential(nn.Dropout2d(p=0.25),
                                      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, time_kernel), padding=0),
                                      nn.BatchNorm2d(32),
                                      nn.ELU(True))
        self.conv3 = nn.Sequential(nn.Dropout2d(p=0.25),
                                      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, time_kernel), padding=0),
                                      nn.BatchNorm2d(64),
                                      nn.ELU(True))
        self.spatial = nn.Sequential(nn.Conv2d(16, 16, (channels, 1), stride=1, padding=0),
                                     nn.BatchNorm2d(16))

        self.maxpool = nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel), padding=0)
        #self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.gconv1 = GraphConv(num_node_feat, hidden_channels).double()
        self.gconv2 = GraphConv(hidden_channels, hidden_channels).double()
        self.avgpool = nn.AvgPool2d(kernel_size=(1,20))

    
        self.lin = nn.Linear(features, num_classes)
        self.act = nn.ELU()

    def forward(self, x, edge_index, edge_atrr, batch, prt):
        size = len(prt)
        x_conv = torch.zeros(size-1,1,self.channels, self.timepoints).to(self.device)
   
        gcnn_out = self.gconv1(x, edge_index, edge_atrr)
        gcnn_out = self.act(gcnn_out)
        gcnn_out = self.gconv2(gcnn_out, edge_index, edge_atrr)
        gcnn_out = self.act(gcnn_out)
        gcnn_out = global_mean_pool(gcnn_out, batch)  # [batch_size, hidden_channels

        # Reaarange tensor for CNN layers
        for i in range(len(prt) - 1):
            x_conv[i,0,:, :] = x[prt[i]:prt[i + 1], :]
        cnn_out= self.conv1 (x_conv)
        cnn_out = self.spatial(cnn_out)
        cnn_out = self.conv2(cnn_out)
        cnn_out = self.maxpool(cnn_out)
        cnn_out = self.conv3(cnn_out)
        cnn_out = self.maxpool(cnn_out)
        cnn=cnn_out.view(cnn_out.size(0),-1)
        
        cat_out= torch.cat((cnn, gcnn_out),1)
        out_3 = self.lin(cat_out)
        return out_3, cat_out
