import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.unet import UNetResidualDualPath
from data.dataset import DataModule, H5FileManager

def main():
    parser = argparse.ArgumentParser(description='Train U-Net with Dual-Path processing for mel spectrogram reconstruction')
    parser.add_argument('--config', type=str, default='config/dual_path_model.yaml', help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None, help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None, help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (overrides config)')
    parser.add_argument('--disable_dual_path', action='store_true', help='Disable dual-path blocks')
    parser.add_argument('--disable_nl', action='store_true', help='Disable non-local blocks')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
        
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    
    if args.h5_path:
        h5_dir = os.path.dirname(args.h5_path)
        h5_file = os.path.basename(args.h5_path)
        config['data']['bin_dir'] = h5_dir
        config['data']['bin_file'] = h5_file
    
    if args.data_key:
        config['data']['data_key'] = args.data_key
    
    if args.max_samples:
        config['data']['max_samples'] = args.max_samples
    
    if 'dual_path' not in config['model']:
        config['model']['dual_path'] = {}
    
    if args.disable_dual_path:
        config['model']['dual_path']['use_dual_path'] = False
        
    if args.disable_nl:
        config['model']['nl_blocks']['use_nl_blocks'] = False
    
    save_dir = args.save_dir or config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs'
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='unet-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min'
        ),
    ]
    
    model = UNetResidualDualPath(config)
    data_module = DataModule(config)
    
    trainer = pl.Trainer(
        max_epochs=config['train']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['train'].get('log_interval', 10),
        deterministic=True,
        accelerator='auto',
        devices='auto',
        precision='32-true'
    )
    
    try:
        trainer.fit(model, data_module, ckpt_path=args.resume)
        print(f"Training completed. Best model saved with val_loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")
    finally:
        H5FileManager.get_instance().close_all()
    
if __name__ == "__main__":
    main()