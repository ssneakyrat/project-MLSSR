import numpy as np
from models.noise_strategies.base import NoiseStrategy

class CyclicalNoiseStrategy(NoiseStrategy):
    """
    Strategy that cycles noise levels throughout training.
    
    This approach oscillates the noise level between min and max values,
    which may help escape local minima, similar to cyclical learning rates.
    """
    def __init__(self, config):
        super().__init__(config)
        self.cycle_steps = config['conditioning'].get('cycle_steps', 1000)
        self.min_noise_level = config['conditioning'].get('min_noise_level', 0.0)
        self.max_noise_level = config['conditioning'].get('max_noise_level', 1.0)
        self.cycle_mode = config['conditioning'].get('cycle_mode', 'cosine')
    
    def get_noise_level(self, global_step):
        """
        Calculate noise level based on global training step in a cyclical pattern
        
        Args:
            global_step: Current global step in training
            
        Returns:
            Float representing cyclical noise level between min_noise_level and max_noise_level
        """
        # Calculate position within cycle (0-1)
        cycle_position = (global_step % self.cycle_steps) / self.cycle_steps
        
        if self.cycle_mode == 'cosine':
            # Cosine wave between min and max noise level
            amplitude = (self.max_noise_level - self.min_noise_level) / 2
            offset = self.min_noise_level + amplitude
            return offset + amplitude * np.cos(2 * np.pi * cycle_position)
        else:  # triangular
            # Triangular wave between min and max noise level
            if cycle_position < 0.5:
                # Rising phase
                return self.min_noise_level + (self.max_noise_level - self.min_noise_level) * (2 * cycle_position)
            else:
                # Falling phase
                return self.max_noise_level - (self.max_noise_level - self.min_noise_level) * (2 * (cycle_position - 0.5))