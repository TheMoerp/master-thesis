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
                        spatial_dims=3,  # Changed from dimensions to spatial_dims
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
        # Store features for skip connections
        features = []
        
        # Initial convolution
        x = self.initial_conv(x)
        features.append(x)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
        
        return x, features

class Decoder(nn.Module):
    def __init__(self, out_channels=1, features=[256, 128, 64, 32, 16]):
        super(Decoder, self).__init__()
        
        self.decoder_blocks = nn.ModuleList()
        
        # Decoder blocks
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
                        spatial_dims=3,  # Changed from dimensions to spatial_dims
                        in_channels=features[i+1] * 2,  # *2 for skip connections
                        out_channels=features[i+1],
                        strides=1,
                        kernel_size=3,
                        subunits=2,
                        act=Act.PRELU,
                        norm=Norm.INSTANCE,
                    )
                )
            )
        
        # Final convolution
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
        # Decoder blocks with skip connections
        for i, block in enumerate(self.decoder_blocks):
            # First part: upsampling
            x = block[0](x)
            
            # Skip connection - use encoder features in reverse order
            skip_features = encoder_features[-(i+2)]  # -2 to skip the bottleneck
            
            # Concatenate along channel dimension
            x = torch.cat([x, skip_features], dim=1)
            
            # Second part: refinement
            x = block[1](x)
            
        # Final convolution
        x = self.final_conv(x)
        
        return x

class AutoEncoder3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256]):
        super(AutoEncoder3D, self).__init__()
        
        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(out_channels, features[::-1])
        
    def forward(self, x):
        # Encoding
        bottleneck, encoder_features = self.encoder(x)
        
        # Decoding
        reconstruction = self.decoder(bottleneck, encoder_features)
        
        return reconstruction
    
    def get_reconstruction_error(self, x):
        """
        Calculate the reconstruction error for anomaly detection.
        
        Args:
            x: Input tensor
            
        Returns:
            reconstruction: Reconstructed output
            error_map: Pixel-wise reconstruction error
            error_score: Overall reconstruction error score
        """
        # Get reconstruction
        reconstruction = self.forward(x)
        
        # Calculate pixel-wise error map (L1 distance)
        error_map = torch.abs(x - reconstruction)
        
        # Calculate overall error score (mean squared error)
        error_score = F.mse_loss(reconstruction, x, reduction='none')
        error_score = error_score.mean(dim=[1, 2, 3, 4])  # Mean over all dimensions except batch
        
        return reconstruction, error_map, error_score

# Define anomaly threshold calculator
class AnomalyDetector:
    def __init__(self, model, threshold_percentile=95):
        """
        Anomaly detector based on reconstruction error.
        
        Args:
            model: Trained autoencoder model
            threshold_percentile: Percentile for threshold calculation
        """
        self.model = model
        self.threshold_percentile = threshold_percentile
        self.threshold = None
    
    def calibrate(self, val_loader, device='cuda'):
        """
        Calibrate the anomaly threshold based on validation data.
        
        Args:
            val_loader: Validation data loader with normal samples
            device: Device to use for computation
        """
        self.model.eval()
        error_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                
                # Get reconstruction error
                _, _, error_score = self.model.get_reconstruction_error(images)
                error_scores.append(error_score.cpu().numpy())
        
        # Concatenate all error scores
        all_errors = np.concatenate(error_scores)
        
        # Calculate threshold
        self.threshold = np.percentile(all_errors, self.threshold_percentile)
        
        return self.threshold
    
    def detect(self, images, device='cuda'):
        """
        Detect anomalies in images.
        
        Args:
            images: Input images tensor
            device: Device to use for computation
            
        Returns:
            is_anomaly: Boolean tensor indicating whether each image is an anomaly
            error_map: Pixel-wise error map
            error_score: Overall error score
        """
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(device)
            
            # Get reconstruction error
            reconstruction, error_map, error_score = self.model.get_reconstruction_error(images)
            
            # Check if error score exceeds threshold
            is_anomaly = error_score > self.threshold
            
        return is_anomaly, error_map, error_score, reconstruction 