class NoiseStrategy:
    """Base interface for noise application strategies"""
    def __init__(self, config):
        self.config = config
    
    def get_noise_level(self, global_step):
        """
        Returns noise level (0-1) based on training progress
        
        Args:
            global_step: Current global step in training
            
        Returns:
            Float representing noise level between 0.0 and 1.0
        """
        raise NotImplementedError("Each noise strategy must implement get_noise_level method")