from models.noise_strategies.base import NoiseStrategy

class NoiseStepStrategy(NoiseStrategy):
    """
    Strategy that linearly increases noise over training.
    
    This is the default strategy that gradually transitions from clean
    reconstruction to full noise conditioning.
    """
    def __init__(self, config):
        super().__init__(config)
        self.noise_ramp_steps = config['conditioning'].get('noise_ramp_steps', 10000)
        self.min_noise_level = config['conditioning'].get('min_noise_level', 0.0)
    
    def get_noise_level(self, global_step):
        """
        Calculate noise level based on global training step
        
        Args:
            global_step: Current global step in training
            
        Returns:
            Float representing noise level between min_noise_level and 1.0
        """
        if self.noise_ramp_steps <= 0:
            return 1.0
        
        progress = min(1.0, global_step / self.noise_ramp_steps)
        return self.min_noise_level + (1.0 - self.min_noise_level) * progress