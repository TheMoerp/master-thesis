import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_channels=1, features=[16, 32, 64, 128, 256]):
        super(Encoder, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=features[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.InstanceNorm3d(features[0]),
            nn.PReLU()
        )
        
        # Encoder blocks
        for i in range(len(features) - 1):
            self.encoder_blocks.append(
                nn.Sequential(
                    ResidualUnit(
                        spatial_dims=3,
                        in_channels=features[i],
                        out_channels=features[i],
                        strides=1,
                        kernel_size=3,
                        subunits=2,
                        act=Act.PRELU,
                        norm=Norm.INSTANCE,
                    ),
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=features[i],
                            out_channels=features[i+1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.InstanceNorm3d(features[i+1]),
                        nn.PReLU()
                    )
                )
            )
    
    def forward(self, x):
        features = []
        x = self.initial_conv(x)
        features.append(x)
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
        return x, features

class Decoder(nn.Module):
    def __init__(self, out_channels=1, features=[256, 128, 64, 32, 16]):
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels=features[i],
                        out_channels=features[i+1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    ResidualUnit(
                        spatial_dims=3,
                        in_channels=features[i+1] * 2,
                        out_channels=features[i+1],
                        strides=1,
                        kernel_size=3,
                        subunits=2,
                        act=Act.PRELU,
                        norm=Norm.INSTANCE,
                    )
                )
            )
        self.final_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=features[-1],
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
    
    def forward(self, x, encoder_features):
        for i, block in enumerate(self.decoder_blocks):
            x_upsampled = block[0](x)
            x = x_upsampled
            skip_features = encoder_features[-(i+2)]
            
            # Ensure spatial dimensions match before concatenation
            if x.shape[2:] != skip_features.shape[2:]:
                skip_features = F.interpolate(skip_features, size=x.shape[2:], mode='trilinear', align_corners=False)
                
            x = torch.cat([x, skip_features], dim=1)
            x = block[1](x)
        x = self.final_conv(x)
        return x

class AutoEncoder3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256]):
        super(AutoEncoder3D, self).__init__()
        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(out_channels, features[::-1])
        
    def forward(self, x):
        input_size = x.shape[2:]
        bottleneck, encoder_features = self.encoder(x)
        reconstruction = self.decoder(bottleneck, encoder_features)
        
        # Ensure reconstruction has the same spatial dimensions as the input
        if reconstruction.shape[2:] != input_size:
            reconstruction = F.interpolate(reconstruction, size=input_size, mode='trilinear', align_corners=False)
            
        return reconstruction
    
    def get_reconstruction_error(self, x):
        reconstruction = self.forward(x)
        error_map = torch.abs(x - reconstruction)
        error_score = F.mse_loss(reconstruction, x, reduction='none')
        error_score = error_score.mean(dim=[1, 2, 3, 4])
        return reconstruction, error_map, error_score

class AnomalyDetector:
    def __init__(self, model, threshold_percentile=95):
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def calibrate(self, val_loader, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        error_scores = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                _, _, error_score = self.model.get_reconstruction_error(images)
                error_scores.append(error_score.cpu().numpy())
        all_errors = np.concatenate(error_scores)
        self.threshold = np.percentile(all_errors, self.threshold_percentile)
        return self.threshold
    
    def detect(self, images, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        with torch.no_grad():
            images = images.to(device)
            reconstruction, error_map, error_score = self.model.get_reconstruction_error(images)
            is_anomaly = error_score > self.threshold
        return is_anomaly, error_map, error_score, reconstruction 