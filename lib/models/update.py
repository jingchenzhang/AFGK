import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_linear import MultiLinear


import torch
import torch.nn as nn

parent_indices = [
    -1,  # Pelvis (0)
    0,   # Left Hip (1)
    0,   # Right Hip (2)
    0,   # Spine 1 (3)
    1,   # Left Knee (4)
    2,   # Right Knee (5)
    3,   # Spine 2 (6)
    4,   # Left Ankle (7)
    5,   # Right Ankle (8)
    6,   # Spine 3 (9)
    7,   # Left Foot (10)
    8,   # Right Foot (11)
    9,   # Neck (12)
    9,   # Left Collar (13)
    9,   # Right Collar (14)
    12,  # Head (15)
    13,  # Left Shoulder (16)
    14,  # Right Shoulder (17)
    16,  # Left Elbow (18)
    17,  # Right Elbow (19)
    18,  # Left Wrist (20)
    19,  # Right Wrist (21)
    20,  # Left Hand (22)
    21   # Right Hand (23)
]

class GRU(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=448):
        super(GRU, self).__init__()
        self.lz = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.lr = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.lq = nn.Linear(hidden_dim + input_dim, hidden_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)  
 
        z = torch.sigmoid(self.lz(hx))  
        r = torch.sigmoid(self.lr(hx))  
        q = torch.tanh(self.lq(torch.cat([r * h, x], dim=-1)))  

        h = (1 - z) * h + z * q  
        return h
    
class GRU1(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=448):
        super(GRU1, self).__init__()
        self.lz = nn.Linear(hidden_dim *2+ input_dim ,hidden_dim)
        self.lr = nn.Linear(hidden_dim *2+ input_dim, hidden_dim)
        self.lq = nn.Linear(hidden_dim *2+ input_dim, hidden_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)  
        z = torch.sigmoid(self.lz(hx))  
        r = torch.sigmoid(self.lr(hx))  
        q = torch.tanh(self.lq(torch.cat([r * h, x], dim=-1)))  

        h = (1 - z) * h + z * q  
        return h

def make_linear_layers(layer_dims, relu_final=True):
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2 or relu_final:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class PoseGRU(nn.Module):
    def __init__(self, joint_num=24, hidden_dim=32,input_dim=448):
        super(PoseGRU, self).__init__()
        self.grus = nn.ModuleList([GRU( hidden_dim,input_dim) for _ in range(joint_num)])
        self.gruh = nn.ModuleList([GRU( hidden_dim,input_dim) for _ in range(joint_num)])
        self.gru = GRU( hidden_dim,input_dim) 
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.joint_num = joint_num
        self.parent_indices = parent_indices

    def forward(self, hpose, x):
        batch_size = x.size(0)
        x = x[:, None, :].repeat(1, self.joint_num, 1)  
        for i in range(self.joint_num):
            parent_idx = self.parent_indices[i]
            hup=hpose.clone()
            if parent_idx == -1:
                hpose[:, i, :] = self.gru(hpose[:,i, :].clone(), x[:, i, :])
            else:
                hpose[:, i, :] = self.grus[i](hpose[:,parent_idx, :].clone(), x[:, i, :]) 
        return hpose


class UpdateBlock(nn.Module):    
    def __init__(self, input_dim=256, hidden_dim=32):
        super(UpdateBlock, self).__init__()

        self.pose_gru = PoseGRU(24, hidden_dim, input_dim)
        self.shape_gru = GRU(hidden_dim, input_dim)
        self.cam_gru = GRU(hidden_dim, input_dim)

    def forward(self, hpose, hshape, hcam, 
                    loc, pose, shape, cam):

        x = torch.cat([loc, pose, shape, cam], dim=-1)  

        hpose = self.pose_gru(hpose, x)
        hshape = self.shape_gru(hshape, x)
        hcam = self.cam_gru(hcam, x)

        return hpose, hshape, hcam


class Regressor(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, num_layer=1, pose_dim=6):
        super(Regressor, self).__init__()
        input_dim = input_dim + 3

        self.p = self._make_multilinear(num_layer, 24, input_dim, hidden_dim)
        self.s = self._make_linear(num_layer, input_dim, hidden_dim)
        self.c = self._make_linear(num_layer, input_dim, hidden_dim)

        self.decpose = MultiLinear(24, hidden_dim, pose_dim)
        self.decshape = nn.Linear(hidden_dim, 10)
        self.deccam = nn.Linear(hidden_dim, 3)

    def forward(self, hpose, hshape, hcam, bbox_info):
        BN = hpose.shape[0]

        hpose = torch.cat([hpose, bbox_info.unsqueeze(1).repeat(1,24,1)], -1)
        hshape = torch.cat([hshape, bbox_info], -1)
        hcam = torch.cat([hcam, bbox_info], -1)
        
        d_pose = self.decpose(self.p(hpose)).view(BN, -1)
        d_shape = self.decshape(self.s(hshape))
        d_cam = self.deccam(self.c(hcam))

        return d_pose, d_shape, d_cam
    
    def _make_linear(self, num, input_dim, hidden_dim):
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [nn.Linear(plane, hidden_dim), 
                     nn.ReLU(inplace=True)]
            layers.extend(layer)  

            plane = hidden_dim

        return nn.Sequential(*layers)
    
    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [MultiLinear(n_head, plane, hidden_dim), 
                     nn.ReLU(inplace=True)]
            layers.extend(layer)
            
            plane = hidden_dim

        return nn.Sequential(*layers) 
