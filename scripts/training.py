import sys
import argparse
from pathlib import Path
import logging
import warnings
from typing import Optional, Tuple, List, Dict
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,LoadImaged, EnsureChannelFirstd,ConcatItemsd,DeleteItemsd,Orientationd,
    RandSpatialCropd, RandFlipd, RandScaleIntensityd,RandShiftIntensityd,Resized,
    NormalizeIntensityd,RandGaussianNoised,
    Activationsd, AsDiscreted, MapLabelValued
)
from monai.networks.nets import SegResNet
from monai.data import Dataset, DataLoader
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler, TensorBoardStatsHandler,TensorBoardImageHandler,EarlyStopHandler,
    ValidationHandler, CheckpointSaver, MeanDice, LrScheduleHandler
)
from monai.inferers import SimpleInferer,SlidingWindowInferer
from monai.handlers.utils import from_engine
from monai.utils import set_determinism


# Suppress MONAI warnings related to inferers
warnings.filterwarnings(
    "ignore",
    module="monai.inferers.utils"
)

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("tumor_training")

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger  # prevent duplicate handlers

    logger.setLevel(logging.INFO)

    # Define logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training.log"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler 
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    # Silence external libraries
    logging.getLogger("monai").setLevel(logging.WARNING)
    logging.getLogger("ignite").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger

logger = setup_logger()

class TumorSegmentationTraining:

    def __init__(
    self,
    root_dir: str | Path,
    ) -> None:
        """
        Initialize the brain tumor segmentation training workflow.
        Sets up model, loss, optimizer, metrics, and output directories.
        
        Args:
            root_dir: Root directory containing the dataset and where outputs/checkpoints will be saved.
        """

        # Ensure deterministic behavior for reproducibility
        set_determinism(seed=123)

        # Path to root directory
        root_dir = Path(root_dir)

        # Path to dataset directory containing BraTS2021 Training subdirectory
        self.dataset_dir = root_dir / "BraTS2021" / "Training"

        # Path to save training outputs
        self.output_dir = root_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Path to save model checkpoints
        self.chkpt_dir = root_dir / "models"
        self.chkpt_dir.mkdir(parents=True, exist_ok=True)

        # Initialize 3D SegResNet model for tumor segmentation
        self.model = SegResNet(
            spatial_dims=3,
            in_channels=4,   # 4 MRI sequences: FLAIR, T1, T1CE, T2
            out_channels=1,  # binary segmentation mask
            init_filters=16,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            dropout_prob=0.2
        )

        # Dice loss with squared predictions and sigmoid activation
        self.loss = DiceLoss(
            smooth_dr=1e-5,
            smooth_nr=0,
            sigmoid=True,
            squared_pred=True,
        )

        # Metric: Mean Dice including background
        self.key_metric = MeanDice(
            include_background=True,
            reduction="mean",
            output_transform=from_engine(["pred", "label"]),
        )
        
        # Optimizer: Adam with weight decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )

        logger.info("Brain tumor segmentation model initialized...")
        
    
    def create_datalist(
            self, val_size: float = 0.2
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Prepare train/validation datalist from the dataset.
        Verifies all modalities exist and are matched, then splits into training and validation sets.
        
        Args:
            val_size: Fraction of dataset to use for validation (default 0.2).
        
        Returns:
            train_datalist: List of dictionaries with paths for each training patient (flair, t1ce, t2, t1, label).
            val_datalist: List of dictionaries with paths for each validation patient (flair, t1ce, t2, t1, label).
        """

        # Gather folder paths for all MRI sequences
        flair_dir = self.dataset_dir/'FLAIR'
        t1ce_dir = self.dataset_dir/'T1CE'
        t2_dir = self.dataset_dir/'T2'
        t1_dir = self.dataset_dir/'T1' 
        label_dir = self.dataset_dir/'Labels'
        
        # Sort images to maintain correspondence across modalities
        flairs = sorted(flair_dir.glob('*.nii.gz'))
        t1ces = sorted(t1ce_dir.glob('*.nii.gz'))
        t2s = sorted(t2_dir.glob('*.nii.gz'))
        t1s = sorted(t1_dir.glob('*.nii.gz'))
        labels = sorted(label_dir.glob('*.nii.gz'))

        # Raise error if any modality is missing
        if not flairs:
            raise FileNotFoundError(f"No images found in {flair_dir}")
        if not t1ces:
            raise FileNotFoundError(f"No images found in {t1ce_dir}")
        if not t2s:
            raise FileNotFoundError(f"No images found in {t2_dir}")
        if not t1s:
            raise FileNotFoundError(f"No images found in {t1_dir}")
        if not labels:
            raise FileNotFoundError(f"No labels found in {label_dir}")
        
        # Check all modalities have the same number of images
        if len(flairs) != len(labels):
            raise ValueError(f"Mismatched files: {len(flairs)} images vs {len(labels)} labels")
        if len(t1ces) != len(labels):
            raise ValueError(f"Mismatched files: {len(t1ces)} images vs {len(labels)} labels") 
        if len(t2s) != len(labels):
            raise ValueError(f"Mismatched files: {len(t2s)} images vs {len(labels)} labels")
        if len(t1s) != len(labels):
            raise ValueError(f"Mismatched files: {len(t1s)} images vs {len(labels)} labels") 
        
        # Create dictionary for each subject containing paths to all modalities
        data_dict = [
            {'flair': str(flair), 't1ce': str(t1ce), 't2': str(t2), 't1': str(t1), 'label': str(lbl)} 
            for flair, t1ce, t2, t1, lbl in zip(flairs, t1ces, t2s, t1s, labels)
        ]
    
        # Train/validation split
        indices = list(range(len(data_dict)))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,  
            random_state=42,
            shuffle=True
        )

        # Create datalist from dictionary        
        train_datalist = [data_dict[i] for i in train_idx]
        val_datalist = [data_dict[i] for i in val_idx]
    
        return train_datalist, val_datalist
        
    def load_pretrained_weights(
            self, pretrained_path: str,
            freeze_encoder: Optional[bool] = False
        ) -> None:
        """
        Load pretrained model weights and optionally freeze the encoder.
        Adapts final output layer if output channels mismatch.
        
        Args:
            pretrained_path: Path to pretrained model checkpoint.
            freeze_encoder: Whether to freeze encoder layers (default False).
        """

        # Load weights from pretrained checkpoint, adapt output layer if needed
        
        path = Path(pretrained_path)

        if path.exists():

            logger.info(f"Loading pretrained weights from {path.name}")
            
            # Load the pretrained state dict
            pretrained_dict = torch.load(path, map_location=torch.device('cpu')) # Adjust if your checkpoint structure differs
            model_dict = self.model.state_dict()

            # Adapt final layer weights if output channels mismatch
            out_weights = pretrained_dict['conv_final.2.conv.weight'] 
            out_bias = pretrained_dict['conv_final.2.conv.bias']  
            adapted_out_weights = out_weights.mean(dim=0, keepdim=True)
            adapted_out_bias = out_bias.mean(dim=0, keepdim=True)   

            pretrained_dict['conv_final.2.conv.weight'] = adapted_out_weights
            pretrained_dict['conv_final.2.conv.bias'] = adapted_out_bias
                
            # Filter out unnecessary keys (mismatched layers)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                if k in model_dict and v.shape == model_dict[k].shape}
                
            # Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
                
            # Load the modified state dict
            self.model.load_state_dict(model_dict, strict=False)  
            
            logger.info(
                f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers "
                f"({len(pretrained_dict)/len(model_dict):.1%})"
            )
            
            # Freeze encoder if requested
            if freeze_encoder:
                self.freeze_encoder()

        else:
            raise FileNotFoundError(f"Pretrained weights not found in: {path.name}")
        

    def freeze_encoder(self) -> None:
        """
        Freeze encoder layers of the SegResNet model, leaving decoder and final layer trainable.
        Logs statistics of total parameters and trainable layers.
        """

        logger.info("Freezing encoder layers")

        # Freeze encoder
        for param in self.model.convInit.parameters():
            param.requires_grad = False

        for param in self.model.down_layers.parameters():
            param.requires_grad = False

        # Ensure decoder + head trainable
        for param in self.model.up_samples.parameters():
            param.requires_grad = True

        for param in self.model.up_layers.parameters():
            param.requires_grad = True

        for param in self.model.conv_final.parameters():
            param.requires_grad = True

        # Parameter statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(
            f"Trainable parameters: {trainable_params}/{total_params} "
            f"({trainable_params/total_params:.1%})"
        )

        # Convolution layer statistics
        total_convs = 0
        frozen_convs = 0
        trainable_convs = 0

        for _ , module in self.model.named_modules():
            if isinstance(module, nn.Conv3d):
                total_convs += 1

                if all(not p.requires_grad for p in module.parameters()):
                    frozen_convs += 1
                else:
                    trainable_convs += 1

        logger.info(
            f"Trainable layers: {trainable_convs}/{total_convs} "
            f"({trainable_convs/total_convs:.1%})"
        )
               
    def train(
        self,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        max_epochs: int = 100,
        batch_size: int = 2,
    ) -> None:
        """
        Build and run the full training pipeline for brain tumor segmentation.
        Sets up preprocessing, data loaders, trainer/evaluator engines, metrics, checkpointing, and early stopping.
        Optionally loads pretrained weights and freezes the encoder.
        
        Args:
            pretrained_path: Path to pretrained weights to load (default None).
            freeze_encoder: Whether to freeze encoder layers when loading pretrained weights (default False).
            max_epochs: Maximum number of training epochs (default 100).
            batch_size: Mini-batch size for training (default 2).
        """

        logger.info("Building training pipeline...") 

        # Generate dataset list for training and validation
        train_datalist, val_datalist = self.create_datalist()
        
        # Compose preprocessing pipeline for training and validation

        # Deterministic transforms for both training and validation
        det_transforms = Compose([
            LoadImaged(keys=["flair", "t1ce", "t2", "t1", "label"], image_only=False),
            EnsureChannelFirstd(keys=["flair", "t1ce", "t2", "t1", "label"]),
            ConcatItemsd(keys=["flair", "t1ce", "t2", "t1"], name="image", dim=0),
            DeleteItemsd(keys=["flair", "t1ce", "t2", "t1"]),
            MapLabelValued(keys="label",orig_labels=[0, 1, 2, 4],target_labels=[0, 1, 1, 1]),
            Orientationd(keys=["image", "label"], axcodes="LPS",labels=None),
            Resized(keys=["image", "label"], spatial_size=(240, 240, 155), mode=("bilinear", "nearest"), align_corners=True),
            NormalizeIntensityd(keys=["image"],nonzero=True,channel_wise=True)
        ])

        # Random transforms for training only
        rand_transforms = Compose([
                RandSpatialCropd(keys=["image", "label"], roi_size=(244,244,144), random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandScaleIntensityd(keys=["image"], factors=0.1, prob=1),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1),
                RandGaussianNoised(keys=["image"],mean=0,std=0.1,prob=0.5),
            ])
        
        # Compose postprocessing pipeline for predictions        
        postprocessing = Compose([
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ])
        
        train_preprocessing = Compose([det_transforms, rand_transforms])
        val_preprocessing = det_transforms

        # Create dataset and dataloader for training and validation
        train_dataset = Dataset(data=train_datalist, transform=train_preprocessing)
        val_dataset = Dataset(data=val_datalist, transform=val_preprocessing)
        
        logger.info(
                    f"Training Dataset | {len(train_dataset)} images"
                    f"Validation Dataset | {len(val_dataset)} images"
                )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )            

        # Inferers for training and validation
        train_inferer = SimpleInferer()
        val_inferer = SlidingWindowInferer(roi_size=(240,240,160),sw_batch_size=1, overlap=0.5)

        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        
        # Set up evaluator engine for validation
        evaluator = SupervisedEvaluator(
        device=torch.device("cpu"),
        val_data_loader=val_dataloader,
        network=self.model,
        inferer=val_inferer,
        key_val_metric={"val_mean_dice":self.key_metric},
        postprocessing=postprocessing,
        val_handlers= [
            StatsHandler(tag_name="val_loss", iteration_log=False),
            TensorBoardStatsHandler(log_dir=self.output_dir, iteration_log=False, tag_name="val_loss", output_transform=from_engine(['loss'], first=True)),
            TensorBoardImageHandler(
            log_dir=self.output_dir,
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
            CheckpointSaver(
                save_dir=self.chkpt_dir,
                save_dict={"model": self.model},
                save_key_metric=True,
                key_metric_filename="model.pt",
                )
            ] ,
        amp=True if torch.cuda.is_available() else False       
        )   
        
        # Set up trainer engine for training
        trainer = SupervisedTrainer(
        device=torch.device("cpu"),
        max_epochs=max_epochs,
        train_data_loader=train_dataloader,
        network=self.model,
        optimizer=self.optimizer,
        loss_function=self.loss,
        inferer=train_inferer,
        postprocessing=postprocessing,
        key_train_metric={"train_mean_dice":self.key_metric},
        train_handlers = [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
            ValidationHandler(validator=evaluator, epoch_level=True, interval=1),
            StatsHandler(tag_name="train_loss", output_transform=from_engine(['loss'], first=True)),
            TensorBoardStatsHandler(log_dir=self.output_dir,tag_name="train_loss", output_transform=from_engine(['loss'], first=True))
        ],
        amp=True if torch.cuda.is_available() else False
        )
        
        # Set up early stopping based on validation mean dice score
        early_stopper = EarlyStopHandler(
            patience=50,
            score_function=lambda x: x.state.metrics["val_mean_dice"],
            epoch_level=True,
            trainer=trainer,
            min_delta=0.01,
            cumulative_delta=True       
        )
        early_stopper.attach(evaluator)  

        logger.info(
            f"Config | device={torch.device('cpu')} | "
            f"epochs={max_epochs} | "
            f"batch_size={batch_size} | "
            f"lr={self.optimizer.param_groups[0]['lr']}"
        )      
                
        # Load pretrained weights
        if pretrained_path is not None:
            self.load_pretrained_weights(pretrained_path, freeze_encoder)
        
        # Start training
        try:
            logger.info("Training started")
            trainer.run()
            logger.info("Training completed successfully")
        except Exception:
            logger.exception("Training crashed")
            raise
               
    
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="MONAI SegResNet training for brain tumor segmentation"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing BraTS2021/Training, models/, and results/"
    )

    parser.add_argument(
        "--pretrained_path",
        type=str,
        help="Path to load pretrained weights (.pt)"
    )

    parser.add_argument(
        "--freeze_encoder",
        type=bool,
        default=False,
        help="Whether to freeze the encoder layers when using pretrained weights"
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )


    args = parser.parse_args()

    # Initialize training workflow
    workflow = TumorSegmentationTraining(
        root_dir=args.root_dir,
    )

    # Run training
    workflow.train(
        pretrained_path=args.pretrained_path,
        freeze_encoder=args.freeze_encoder,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )

    
    