import sys
import argparse
from pathlib import Path
import logging
import warnings
from typing import List, Dict
import torch
from monai.transforms import (
    Compose,LoadImaged, EnsureChannelFirstd,ConcatItemsd,DeleteItemsd,Orientationd,Resized,
    NormalizeIntensityd,Activationsd, AsDiscreted, MapLabelValued,Invertd
)
from monai.networks.nets import SegResNet
from monai.data import Dataset, DataLoader
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    MeanDice, CheckpointLoader,MetricsSaver
)
from monai.inferers import SlidingWindowInferer
from monai.handlers.utils import from_engine
from monai.utils import set_determinism

# Suppress MONAI warnings related to inferers
warnings.filterwarnings(
    "ignore",
    module="monai.inferers.utils"
)

# --------- Logging setup ----------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("tumor_evaluation")

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
    log_file = log_dir / "evaluation.log"

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

class TumorSegmentationEvaluation:
        
    def __init__(
        self,
        root_dir: str | Path,
        ) -> None:
        """
        Initialize the brain tumor segmentation evaluation workflow.
        Sets up model, metrics, dataset paths, and output directory.
        
        Args:
            root_dir: Root directory containing the dataset, model checkpoint, and where results will be saved.
        """

        # Ensure deterministic behavior for reproducibility
        set_determinism(seed=123)

        # Path to root directory
        root_dir = Path(root_dir)

        # Path to dataset directory containing BraTS2021 Testing subdirectory
        self.dataset_dir = root_dir / "BraTS2021" / "Testing"

        # Path to save evaluation results
        self.output_dir = root_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load path for pretrained model
        self.model_path = root_dir / "models" / "model.pt"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path.name}")

        # Initialize 3D SegResNet model for tumor segmentation
        self.model = SegResNet(
                spatial_dims=3,
                in_channels=4,
                out_channels=1,
                init_filters=16,
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                dropout_prob=0.2,
            )
        
        # Metric: Mean Dice including background
        self.key_metric = MeanDice(
                include_background=True,
                reduction="mean",
                output_transform=from_engine(["pred", "label"]),
            )


        logger.info("Brain tumor segmentation model initialized...")
    
    def create_datalist(
            self
    ) -> List[Dict[str, str]]:
        """
        Prepare a datalist from the testing dataset.
        Verifies all MRI modalities and labels exist and are matched, then creates a list of dictionaries
        with paths to all sequences for each patient.
        
        Returns:
            datalist: List of dictionaries, each containing paths for flair, t1ce, t2, t1, and label images.
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
    
        # Create datalist from dictionary
        datalist = [data_dict[i] for i in range(len(data_dict))]
    
        return datalist
        
               
    def eval(
            self,
            batch_size: int = 2
        ) -> None:
        """
        Run evaluation of the trained segmentation model on the testing dataset.
        Sets up preprocessing, dataloader, sliding window inference, postprocessing, metrics, and checkpoint loading.
        
        Args:
            batch_size: Mini-batch size for evaluation (default 2).
        """

        logger.info("Building evaluation pipeline...") 

        # Generate dataset list
        datalist = self.create_datalist()
        
        # Compose preprocessing pipeline for evaluation
        preprocessing = Compose([
            LoadImaged(keys=["flair", "t1ce", "t2", "t1", "label"], image_only=False),
            EnsureChannelFirstd(keys=["flair", "t1ce", "t2", "t1", "label"]),
            ConcatItemsd(keys=["flair", "t1ce", "t2", "t1"], name="image", dim=0),
            DeleteItemsd(keys=["flair", "t1ce", "t2", "t1"]),
            MapLabelValued(keys="label",orig_labels=[0, 1, 2, 4],target_labels=[0, 1, 1, 1]),
            Orientationd(keys=["image", "label"], axcodes="LPS", labels=None),
            Resized(keys=["image", "label"], spatial_size=(240, 240, 155), mode=("bilinear", "nearest"), align_corners=True),
            NormalizeIntensityd(keys=["image"],nonzero=True,channel_wise=True)
        ])

        # Compose postprocessing pipeline for predictions
        postprocessing = Compose([
            Activationsd(keys="pred", sigmoid=True),
            Invertd(keys="pred",
                    transform=preprocessing,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    nearest_interp=False,
                    to_tensor=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ])

        # Create dataset and dataloader
        dataset = Dataset(data=datalist, transform=preprocessing)

        logger.info(f"Evaluation Dataset | {len(dataset)} images")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        # Use sliding window inference for memory-efficient evaluation
        inferer = SlidingWindowInferer(roi_size=(240,240,160), sw_batch_size=1, overlap=0.5)

        # Set up evaluator engine for evaluation
        evaluator = SupervisedEvaluator(
        device=torch.device("cpu"),
        val_data_loader=dataloader,
        network=self.model,
        inferer=inferer,
        key_val_metric={"eval_mean_dice":self.key_metric},
        postprocessing=postprocessing,
        val_handlers=[
            MetricsSaver(save_dir=self.output_dir,
                         metrics=["eval_mean_dice"],
                         ),
            ],
        amp=True if torch.cuda.is_available() else False       
        )   

        # Load model checkpoint and attach to evaluator
        checkpoint_loader = CheckpointLoader(load_path=self.model_path, load_dict={"model": self.model})
        checkpoint_loader.attach(evaluator)

        logger.info(
            f"Config | device={torch.device('cpu')} | "
            f"batch_size={batch_size} | "
            )
        
        # Start evaluation
        try:
            logger.info("Evaluation started")
            evaluator.run()
            logger.info("Evaluation completed successfully")
        except Exception:
            logger.exception("Evaluation crashed")
            raise
               
    
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="MONAI SegResNet evaluation for brain tumor segmentation"
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing BraTS2021/Testing and models/"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )

    args = parser.parse_args()

    # Initialize evaluation workflow
    workflow = TumorSegmentationEvaluation(
        root_dir=args.root_dir,
    )

    # Run evaluation
    workflow.eval(
        batch_size=args.batch_size
    )

    
    