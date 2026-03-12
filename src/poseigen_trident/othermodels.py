import torch
import torch.nn as nn

class BichromBimodal(nn.Module):
    def __init__(self, dna_filters=128, hist_filters=64, dna_lstm=50, hist_lstm=25):
        super(BichromBimodal, self).__init__()
        
        # --- DNA Branch (Input: N, 1, 500, 4) ---
        # We use a kernel that spans the full 4-base width or a subset
        self.dna_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=dna_filters, kernel_size=(15, 4)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # After Conv2d: (N, dna_filters, 486, 1) -> Squeeze/Flatten to (N, 486, dna_filters)
        self.dna_lstm = nn.LSTM(dna_filters, dna_lstm, batch_first=True)
        self.dna_fc = nn.Linear(dna_lstm, 1)

        # --- Histone Branch (Input: N, 11, 40, 1) ---
        # Note: input channels is 11, we treat tracks as "height" or "channels"
        # Using Conv2d here allows kernels to look across different histones
        self.hist_conv = nn.Sequential(
            nn.Conv2d(in_channels=11, out_channels=hist_filters, kernel_size=(5, 1)),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # After Conv2d: (N, hist_filters, 36, 1) -> Squeeze/Flatten to (N, 36, hist_filters)
        self.hist_lstm = nn.LSTM(hist_filters, hist_lstm, batch_first=True)
        self.hist_fc = nn.Linear(hist_lstm, 1)

        self.sigmoid = nn.Sigmoid()

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
        return self.sigmoid(v_seq + v_hist)
    


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