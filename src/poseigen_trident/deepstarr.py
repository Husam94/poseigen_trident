import numpy as np

import torch
import torch.nn as nn

'''Pytorch implementation of DeepSTARR that is fluent with Trident formatting'''

class DeepSTARR(nn.Module):

    def __init__(self, dim_i = (1, 1000, 4), 
                        dim_o = (1, 1, 1), 
                        ck_i = 7, ck_other = [3, 5, 3, 3, 3],
                        cf_i = 256, cf_other = [60, 60, 120, 120, 120], 
                        num_convs = 6,
                        num_dense = 2, 
                        activations = nn.ReLU(), activation_f = None,
                        dropout = 0.4):
        super(DeepSTARR, self).__init__()

        cks = [ck_i] + ck_other
        cfs = [dim_i[0], cf_i] + cf_other

        conv_layers = []
        for i in range(num_convs):
            wi = 4 if i == 0 else 1
            convpad = cks[i] // 2
            conv_layers.append(nn.Conv2d(cfs[i], cfs[i+1], (cks[i], wi), padding=(convpad, 0), 
                                            bias=True, stride=(1, 1)))
            conv_layers.append(nn.BatchNorm2d(cfs[i+1]))
            conv_layers.append(activations)
            conv_layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1),
                                            padding=0, ceil_mode=True))

        self.conv = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, dim_i[0], dim_i[1], dim_i[2])
            conv_out = self.conv(dummy)
            self.length_afterconv = conv_out.shape[2]   # must be axis 2, not 3

        dense_layers = []
        tot_dense = num_dense + 1
        for i in range(tot_dense):
            dko = self.length_afterconv if i == 0 else 1
            dfi = cfs[num_convs] if i == 0 else 256
            dfo = dim_o[0] if i == tot_dense - 1 else 256
            dense_layers.append(nn.Conv2d(dfi, dfo, (dko, 1), padding='valid', bias=True))
            if i == tot_dense - 1:
                break
            dense_layers.append(nn.BatchNorm2d(dfo))
            dense_layers.append(activations)
            dense_layers.append(nn.Dropout2d(dropout))
        self.dense = nn.Sequential(*dense_layers)
        self.actf = nn.Identity() if activation_f is None else activation_f

        # sanity check: first dense conv should collapse sequence axis to 1
        first_dense = self.dense[0]
        assert isinstance(first_dense, nn.Conv2d)
        assert first_dense.kernel_size[0] == self.length_afterconv, (
            f"First dense kernel ({first_dense.kernel_size[0]}) != post-conv length ({self.length_afterconv})"
        )    

    def forward(self,x):
        
        x = self.conv(x)
        x = self.dense(x)
        x = self.actf(x)    
       
        return x


DeepSTARR_dict = {'dim_i': [[(1, 249, 4)],'cat'],
                        'ck_i': [[7, 12, 19, 24], 'cat'], 
                        'cf_i': [[32, 64, 128, 256, 512, 1024], 'cat'], 
                        'num_convs': [[1,2,3,4], 'cat'], 
                        'num_dense': [[1,2,3], 'cat']}




def Reset_DeepSTARR(deepstarr):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 

    lconvs, ldens = len(deepstarr.conv), len(deepstarr.dense)

    for i in np.arange(lconvs): 
        if isinstance(deepstarr.conv[i], nn.Conv2d): 
            deepstarr.conv[i].reset_parameters()
    
    for i in np.arange(ldens): 
        if isinstance(deepstarr.dense[i], nn.Conv2d): 
            deepstarr.dense[i].reset_parameters()

    return deepstarr