import torch.nn as nn

class DEPTH_predict_layer(nn.Module):
    def __init__(self, input_dim):
        super(DEPTH_predict_layer, self).__init__()

        self.conv3 = nn.Conv2d(input_dim,1,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv4 = nn.Conv2d(input_dim,1,kernel_size=3,stride=1,padding=1,bias=False)
    
    def forward(self, x):
        return_dict = {}
        x1 = self.conv3(x)
        return_dict['predict_depth'] = x1.squeeze(1)
        x2 = self.conv4(x)
        return_dict['predict_bc'] = x2.squeeze(1)

        return return_dict ,x1