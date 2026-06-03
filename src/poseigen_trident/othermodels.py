import numpy

import torch
import torch.nn as nn

'''Pytorch implementation of other NNs for genomics data'''

class BiChrom(nn.Module):
    def __init__(self, 
                 A_dim_i = (1,500,4),
                 A_kE_k = 24, A_cf = 256, 
                 A_rec_f = 32, 
                 A_dense_num = 3, A_dense_f = 512,
                 
                 B_dim_i = (11,40,1),
                 B_kE_k = 1, B_cf = 15,
                 B_rec_f = 5,
                 
                 rec_layer = 'LSTM', dropout = 0.5, 
                 activation_f = None, activations = nn.ReLU()
                 ):
        super(BiChrom, self).__init__()

        if rec_layer == 'LSTM': rec_lax = torch.nn.LSTM
        elif rec_layer == 'GRU': rec_lax = torch.nn.GRU

        # --- DNA Branch (Input: N, 1, 500, 4) ---
        # We use a kernel that spans the full 4-base width or a subset
        self.dna_conv = nn.Sequential(
            nn.Conv2d(in_channels=A_dim_i[0], out_channels=A_cf, kernel_size=(A_kE_k, A_dim_i[2])),
            nn.ReLU(),
            nn.BatchNorm2d(A_cf),
            nn.MaxPool2d(kernel_size=(15, 1), stride=(15, 1), padding=0, ceil_mode=True)
        )
        # After Conv2d: (N, A_cf, 486, 1) -> Squeeze/Flatten to (N, 486, dna_filter

        self.dna_lstm = rec_lax(A_cf, A_rec_f, batch_first=True)

        lins = []
        for _ in range(A_dense_num):
            lins.extend([nn.Linear(A_rec_f, A_dense_f),
                        activations,
                        nn.Dropout(dropout)])
            A_rec_f = A_dense_f
        lins.append(nn.Linear(A_rec_f, 1))

        self.dna_fc = nn.Sequential(*lins)


        # --- Histone Branch (Input: N, 11, 40, 1) ---
        # Note: input channels is 11, we treat tracks as "height" or "channels"
        # Using Conv2d here allows kernels to look across different histones
        self.hist_conv = nn.Sequential(
            nn.Conv2d(in_channels=B_dim_i[0], out_channels=B_cf, kernel_size=(B_kE_k, B_dim_i[2])),
            activations
        )
        # After Conv2d: (N, hist_filters, 36, 1) -> Squeeze/Flatten to (N, 36, hist_filters)
        self.hist_lstm = rec_lax(B_cf, B_rec_f, batch_first=True)
        self.hist_fc = nn.Linear(B_rec_f, 1)

        self.actf = nn.Identity() if activation_f is None else activation_f

    def forward(self, dna, hist):
        # dna: (N, 1, 500, 4) | hist: (N, 11, 40, 1)

        # 1. DNA Branch Processing
        x_dna = self.dna_conv(dna) # (N, dna_filters, 486, 1)
        x_dna = x_dna.squeeze(-1).permute(0, 2, 1) # (N, 486, dna_filters)
        _, (h_dna, _) = self.dna_lstm(x_dna)
        v_seq = self.dna_fc(h_dna.squeeze(0)) # Scalar Activation

        # 2. Histone Branch Processing
        x_hist = self.hist_conv(hist) # (N, hist_filters, 36, 1)
        x_hist = x_hist.squeeze(-1).permute(0, 2, 1) # (N, 36, hist_filters)
        _, (h_hist, _) = self.hist_lstm(x_hist)
        v_hist = self.hist_fc(h_hist.squeeze(0)) # Scalar Activation

        # 3. Late Fusion
        out = self.actf(v_seq + v_hist)  # (N, 1)
        return out.view(out.size(0), 1, 1, 1)
    

def Reset_BiChrom(bichrom):
    #ds consists of a "conv" and a "dense" module. Need to go through each one, see if its a conv and reset if so. 
    lke, los = len(bichrom.dna_conv), len(bichrom.hist_conv)

    for i in np.arange(lke): 
        if isinstance(bichrom.dna_conv[i], nn.Conv2d): 
            bichrom.dna_conv[i].reset_parameters()
        
    for i in np.arange(los): 
        if isinstance(bichrom.hist_conv[i], nn.Conv2d): 
            bichrom.hist_conv[i].reset_parameters()
    
    for i in np.arange(len(bichrom.dna_fc)): 
        if isinstance(bichrom.dna_fc[i], nn.Linear): 
            bichrom.dna_fc[i].reset_parameters()
    
    bichrom.dna_lstm.reset_parameters()
    bichrom.hist_lstm.reset_parameters()
    bichrom.hist_fc.reset_parameters()

    print('done reset mod')

    return bichrom


import torch
import torch.nn as nn
import torch.nn.functional as F

class iSEGnet(nn.Module):
    def __init__(self, num_epigenetic_tracks=6, dna_length=2500):
        super(iSEGnet, self).__init__()
        
        # --- DNA Sequence Branch ---
        # Input shape: (Batch, 4, dna_length)
        self.dna_conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=128, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Flatten()
        )
        
        # --- Epigenetic Signal Branch ---
        # Input shape: (Batch, num_epigenetic_tracks, dna_length)
        self.epi_conv = nn.Sequential(
            nn.Conv1d(in_channels=num_epigenetic_tracks, out_channels=64, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
            nn.Flatten()
        )
        
        # Calculate flattened sizes for the linear layer
        # For 2500bp: after two poolings of 5, size is 2500 / 5 / 5 = 100
        flattened_dna_dim = 64 * (dna_length // 25)
        flattened_epi_dim = 32 * (dna_length // 25)
        
        # --- Fusion & Fully Connected Layers ---
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_dna_dim + flattened_epi_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Regression output for mRNA abundance
        )

    def forward(self, dna_seq, epi_signal):
        # dna_seq: (Batch, 4, SeqLen)
        # epi_signal: (Batch, Tracks, SeqLen)
        
        dna_features = self.dna_conv(dna_seq)
        epi_features = self.epi_conv(epi_signal)
        
        # Concatenate features from both branches
        combined = torch.cat((dna_features, epi_features), dim=1)
        
        output = self.fc_layers(combined)
        return output

# Example usage:
# model = iSEGnet(num_epigenetic_tracks=6, dna_length=2500)
# dna_input = torch.randn(32, 4, 2500)
# epi_input = torch.randn(32, 6, 2500)
# prediction = model(dna_input, epi_input)