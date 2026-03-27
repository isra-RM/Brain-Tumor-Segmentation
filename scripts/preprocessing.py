from pathlib import Path
import random
import shutil
import logging
import sys
import argparse
from typing import Optional
from tqdm import tqdm

# --------- Logging setup ----------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("tumor_preprocessing")

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
    log_file = log_dir / "preprocessing.log"

    # Console logging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File logging
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


logger = setup_logger()


class BraTSPreprocessing:
   
    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        n_train: int = 400,
        n_test: int = 100,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Initialize preprocessing workflow
        Args:
            source_dir: Original BraTS2021 dataset location
            target_dir: Directory to store processed dataset
            n_train: Number of training patients
            n_test: Number of testing patients
            seed: Random seed for reproducibility
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.n_train = n_train
        self.n_test = n_test
        self.seed = seed

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")

        # Logging basic info
        logger.info("BraTS2021 preprocessing initialized")
        logger.info(f"Source directory : {self.source_dir}")
        logger.info(f"Target directory : {self.target_dir}")
        logger.info(f"Train/Test split : {self.n_train}/{self.n_test}")


    def split_dataset(self) -> None:
        """
        Randomly select patients and copy into Training and Testing folders.
        """
        # List all patient folders starting with "BraTS2021_"
        all_patients = [
            f for f in self.source_dir.iterdir()
            if f.is_dir() and f.name.startswith("BraTS2021_")
        ]

        total_required = self.n_train + self.n_test

        logger.info("Starting dataset split...")
        logger.info(f"Available patients: {len(all_patients)} | Requested: {total_required}")

        if len(all_patients) < total_required:
            raise ValueError(f"Requested {total_required} patients but only {len(all_patients)} available.")

        # Shuffle patients for random selection
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(all_patients)
        logger.info("Patients shuffled successfully")

        # Assign train and test patients
        train_patients = all_patients[:self.n_train]
        test_patients = all_patients[self.n_train:total_required]

        # Copy patient folders to split directories
        for split_name, patients in {"Training": train_patients, "Testing": test_patients}.items():
            split_dir = self.target_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"[{split_name}] Copying {len(patients)} patients → {split_dir}")

            for folder in tqdm(patients, desc=f"Copying {split_name}", leave=False, dynamic_ncols=True):
                destination = split_dir / folder.name
                if destination.exists():
                    raise FileExistsError(f"Destination exists: {destination}")
                shutil.copytree(folder, destination)

        logger.info("Dataset split completed successfully.")


    def organize_modalities(self, delete_after: bool = True) -> None:
        """
        Move patient image files into modality-specific subfolders
        Args:
            delete_after: Remove original patient folders after organizing
        """
        logger.info("Starting modality organization...")

        for split_name in ["Training", "Testing"]:
            split_dir = self.target_dir / split_name
            if not split_dir.exists():
                raise FileNotFoundError(f"{split_name} folder not found.")

            # Create modality folders
            modality_dirs = {
                "t1": split_dir / "T1",
                "t1ce": split_dir / "T1CE",
                "t2": split_dir / "T2",
                "flair": split_dir / "FLAIR",
                "seg": split_dir / "Labels",
            }
            for path in modality_dirs.values():
                path.mkdir(parents=True, exist_ok=True)

            # List all patient folders in current split
            patient_folders = [f for f in split_dir.iterdir() if f.is_dir() and f.name.startswith("BraTS2021_")]
            logger.info(f"[{split_name}] folder | organizing modalities for {len(patient_folders)} patients")

            for patient_folder in tqdm(patient_folders, desc=f"Organizing {split_name}", leave=False, dynamic_ncols=True):
                for file in patient_folder.glob("*.nii*"):
                    # Determine modality key based on filename
                    name_lower = file.name.lower()
                    if "_t1ce" in name_lower:
                        key = "t1ce"
                    elif "_t1." in name_lower or "_t1.nii" in name_lower:
                        key = "t1"
                    elif "_t2" in name_lower:
                        key = "t2"
                    elif "_flair" in name_lower:
                        key = "flair"
                    elif "_seg" in name_lower:
                        key = "seg"
                    else:
                        continue

                    # Move file to corresponding modality folder
                    destination = modality_dirs[key] / file.name
                    if destination.exists():
                        raise FileExistsError(f"File already exists: {destination}")
                    shutil.move(str(file), str(destination))

                # Delete original patient folder if requested
                if delete_after:
                    shutil.rmtree(patient_folder)

            logger.info(f"[{split_name}] modalities organized successfully")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Organize BraTS2021 dataset into Training/Testing."
        )
    
    parser.add_argument(
        "--source_dir", 
        type=str, 
        required=True
        )
    
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True
        )
    
    parser.add_argument(
        "--n_train", 
        type=int, 
        default=400
        )
    parser.add_argument(
        "--n_test", 
        type=int, 
        default=100
        )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42
        )

    args = parser.parse_args()

    # Initialize preprocessing workflow
    preprocessor = BraTSPreprocessing(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
    )

    # Execute dataset split
    preprocessor.split_dataset()
    # Execute modality organization
    preprocessor.organize_modalities(delete_after=True)