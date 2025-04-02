import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss as described in the paper:
    "Neural Speech Synthesis with Transformer Network".
    
    This loss measures the relative error between the magnitudes of the
    target and predicted spectrograms.
    """
    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (Tensor): Predicted spectrogram [B, 1, T, F]
            y_true (Tensor): Target spectrogram [B, 1, T, F]
            
        Returns:
            Tensor: Spectral convergence loss
        """
        # Reshape tensors to apply L2 norm
        batch_size = y_true.shape[0]
        y_true_flat = y_true.view(batch_size, -1)
        y_pred_flat = y_pred.view(batch_size, -1)
        
        # Calculate L2 norm for each flattened spectrogram
        numerator = torch.norm(y_true_flat - y_pred_flat, p=2, dim=1)
        denominator = torch.norm(y_true_flat, p=2, dim=1)
        
        # Avoid division by zero
        loss = numerator / (denominator + 1e-8)
        
        return torch.mean(loss)

class CombinedLoss(nn.Module):
    """
    Combined loss function as specified:
    Loss = α·L1 + β·SpectralConvergence
    
    where α=0.8, β=0.2
    """
    def __init__(self, alpha=0.8, beta=0.2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.spectral_loss = SpectralConvergenceLoss()
    
    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (Tensor): Predicted spectrogram [B, 1, T, F]
            y_true (Tensor): Target spectrogram [B, 1, T, F]
            
        Returns:
            Tensor: Combined loss
        """
        l1 = self.l1_loss(y_pred, y_true)
        sc = self.spectral_loss(y_pred, y_true)
        
        return self.alpha * l1 + self.beta * sc