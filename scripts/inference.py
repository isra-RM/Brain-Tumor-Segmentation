import sys
import argparse
from pathlib import Path
import logging
import warnings
from typing import List, Dict, Union, Optional
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd, DeleteItemsd, Orientationd,
    Invertd, Resized, NormalizeIntensityd, SaveImaged,
    Activationsd, AsDiscreted,
)
from monai.handlers import (
    StatsHandler, CheckpointLoader
)
from monai.networks.nets import SegResNet
from monai.data import Dataset, DataLoader
from monai.engines import SupervisedEvaluator
from monai.inferers import SlidingWindowInferer
from monai.utils import set_determinism


# Suppress MONAI warnings related to inferers
warnings.filterwarnings(
    "ignore",
    module="monai.inferers.utils"
)

# ---------- Logging setup ----------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("tumor_inference")

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Define logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "inference.log"

    # Console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File logging
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    # Silence verbose logging from external libraries
    logging.getLogger("monai").setLevel(logging.WARNING)
    logging.getLogger("ignite").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger


logger = setup_logger()


class TumorSegmentationInference:

    def __init__(self,
                 dataset_dir: str | Path,
                 segm_dir: str | Path,
                 model_path: Optional[Union[str, Path]] = "models/model.pt",
                 ) -> None:
        """
        Initialize the brain tumor segmentation inference workflow.
        Sets up model, dataset paths, output directory, and loads pretrained weights.
        
        Args:
            dataset_dir: Path to the directory containing input MRI sequences (FLAIR, T1, T1CE, T2).
            segm_dir: Directory where predicted segmentations will be saved.
            model_path: Path to pretrained model checkpoint (default 'models/model.pt').
        """
        
        # Ensure deterministic behavior for reproducibility
        set_determinism(seed=123)
        
        # Path to input images
        self.dataset_dir = Path(dataset_dir)

        # Path to save segmentation outputs
        self.segm_dir = Path(segm_dir)
        Path(segm_dir).mkdir(parents=True, exist_ok=True)

        # Load path for pretrained model
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
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

        logger.info("Brain tumor segmentation model initialized...")

    def create_datalist(self) -> List[Dict[str, str]]:
        """
        Prepare a datalist from the input dataset for inference.
        Verifies all MRI modalities exist and are matched, then creates a list of dictionaries
        with paths to all sequences for each subject.
        
        Returns:
            datalist: List of dictionaries, each containing paths for flair, t1, t1ce, and t2 images.
        """

        # Gather folder paths for all MRI sequences
        flair_dir = self.dataset_dir / 'FLAIR'
        t1_dir = self.dataset_dir / 'T1'
        t1ce_dir = self.dataset_dir / 'T1CE'
        t2_dir = self.dataset_dir / 'T2'
        
        # Sort images to maintain correspondence across modalities
        flairs = sorted(flair_dir.glob('*.nii.gz'))
        t1s = sorted(t1_dir.glob('*.nii.gz'))
        t1ces = sorted(t1ce_dir.glob('*.nii.gz'))
        t2s = sorted(t2_dir.glob('*.nii.gz'))

        # Raise error if any modality is missing
        if not flairs:
            raise FileNotFoundError(f"No images found in {flair_dir}")
        if not t1s:
            raise FileNotFoundError(f"No images found in {t1_dir}")
        if not t1ces:
            raise FileNotFoundError(f"No images found in {t1ce_dir}")
        if not t2s:
            raise FileNotFoundError(f"No images found in {t2_dir}")

        # Check all modalities have the same number of images
        if len(flairs) != len(t1s) or len(t1s) != len(t1ces) or len(t1ces) != len(t2s):
            raise ValueError(f"Mismatched files: {len(flairs)} flairs vs {len(t1s)} t1s vs {len(t1ces)} t1ces vs {len(t2s)} t2s")
        
        # Create dictionary for each subject containing paths to all modalities
        data_dict = [
            {'flair': str(flair), 't1': str(t1), 't1ce': str(t1ce), 't2': str(t2)} 
            for flair, t1, t1ce, t2 in zip(flairs, t1s, t1ces, t2s)
        ]

        # Create datalist from dictionary
        datalist = [data_dict[i] for i in range(len(data_dict))]

        return datalist


    def infer(self, 
              batch_size: int = 1) -> None:
        """
        Run inference using the pretrained segmentation model on the input dataset.
        Sets up preprocessing, dataloader, sliding window inference, postprocessing, and checkpoint loading.
        
        Args:
            batch_size: Mini-batch size for inference (default 1).
        """
        
        logger.info("Building inference pipeline...") 

        # Generate dataset list
        datalist = self.create_datalist()

        # Compose preprocessing pipeline for inference
        preprocessing = Compose([
            LoadImaged(keys=["flair", "t1", "t1ce", "t2"], image_only=False),  # Load NIfTI images
            EnsureChannelFirstd(keys=["flair", "t1", "t1ce", "t2"]),  # Ensure channel-first format
            ConcatItemsd(keys=["flair", "t1", "t1ce", "t2"], name="image", dim=0),  # Combine modalities
            DeleteItemsd(keys=["flair", "t1", "t1ce", "t2"]),  # Remove original keys
            Orientationd(keys="image", axcodes="LPS", labels=None),  # Standardize orientation
            Resized(keys="image", spatial_size=(240, 240, 155), mode=("bilinear"), align_corners=True),  # Resize to common shape
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)  # Normalize intensity
        ])

        # Compose postprocessing pipeline for predictions
        postprocessing = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),  # Apply sigmoid to predictions
            Invertd(
                keys="pred",  # Invert preprocessing to match original image space
                transform=preprocessing,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", threshold=0.5),  # Threshold to binary mask
            SaveImaged(
                keys="pred",  # Save predicted mask
                meta_keys="pred_meta_dict",
                output_dir=self.segm_dir,
                output_postfix="seg",
                output_ext=".nii.gz",
                output_dtype="uint8", 
                separate_folder=False, 
                resample=False,
                squeeze_end_dims=True,
            ), 
        ]
        )

        # Create dataset and dataloader
        dataset = Dataset(data=datalist, transform=preprocessing)
        logger.info(f"Inference Dataset | {len(dataset)} images")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        # Use sliding window inference for memory-efficient evaluation
        inferer = SlidingWindowInferer(roi_size=(240, 240, 160), sw_batch_size=1, overlap=0.5)
    
        # Setup evaluator engine
        evaluator = SupervisedEvaluator(
            device=torch.device("cpu"),
            val_data_loader=dataloader,
            network=self.model,
            inferer=inferer,
            postprocessing=postprocessing,
            val_handlers=[StatsHandler(iteration_log=False)],  # Track metrics
            amp=True if torch.cuda.is_available() else False,
        )

        # Load model checkpoint before evaluation
        checkpoint_loader = CheckpointLoader(load_path=self.model_path, load_dict={"model": self.model})
        checkpoint_loader.attach(evaluator)

        logger.info(
            f"Config | device={torch.device('cpu')} | batch_size={batch_size}"
        )

        # Start inference
        try:
            logger.info("Inference started")
            evaluator.run()
            logger.info("Inference completed successfully")
        except Exception:
            logger.exception("Inference crashed")
            raise  # Ensure exceptions propagate


if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="MONAI SegResNet inference for brain tumor segmentation"
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset directory containing images for inference. Must create 4 separate subfolders: FLAIR, T1, T1CE, and T2."
    )

    parser.add_argument(
        "--segm_dir",
        type=str,
        default='./segm_dir',
        help="Output directory to save segmentations"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/model.pt",
        help="Path to brain tumor pretrained model"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )

    args = parser.parse_args()

    # Initialize inference workflow
    workflow = TumorSegmentationInference(
        dataset_dir=args.dataset_dir,
        segm_dir=args.segm_dir,
        model_path=args.model_path,
    )

    # Run inference
    workflow.infer(args.batch_size)