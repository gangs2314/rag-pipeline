"""Kaggle dataset integration for downloading sample documents."""

import os
import shutil
from pathlib import Path
from typing import Optional

from rag_pipeline.config import settings


class KaggleIntegration:
    """Handle Kaggle dataset downloads and integration."""

    def __init__(self):
        """Initialize Kaggle integration."""
        self.datasets_dir = settings.kaggle_datasets_dir

    def download_dataset(
        self,
        dataset_name: str,
        output_dir: Optional[Path] = None,
        force: bool = False,
    ) -> dict:
        """
        Download a dataset from Kaggle.

        Args:
            dataset_name: Kaggle dataset identifier (e.g., "username/dataset-name")
            output_dir: Output directory (uses settings.kaggle_datasets_dir if None)
            force: Force re-download even if exists

        Returns:
            dict with download status
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except ImportError:
            return {
                "status": "error",
                "error": "kaggle library not installed. Install with: pip install kaggle",
            }

        output_dir = Path(output_dir or self.datasets_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_dir = output_dir / dataset_name.replace("/", "_")

        # Check if already downloaded
        if dataset_dir.exists() and not force:
            return {
                "status": "success",
                "message": "Dataset already exists",
                "path": str(dataset_dir),
                "new_download": False,
            }

        try:
            api = KaggleApi()
            api.authenticate()

            print(f"Downloading dataset: {dataset_name}")
            api.dataset_download_files(
                dataset_name,
                path=str(dataset_dir),
                unzip=True,
            )

            return {
                "status": "success",
                "message": f"Downloaded dataset: {dataset_name}",
                "path": str(dataset_dir),
                "new_download": True,
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def list_local_datasets(self) -> list[dict]:
        """List locally downloaded datasets."""
        datasets = []

        if not self.datasets_dir.exists():
            return datasets

        for dataset_path in self.datasets_dir.iterdir():
            if dataset_path.is_dir():
                file_count = len(list(dataset_path.glob("**/*")))
                datasets.append(
                    {
                        "name": dataset_path.name,
                        "path": str(dataset_path),
                        "file_count": file_count,
                    }
                )

        return datasets

    def get_sample_documents_dir(self, dataset_name: str) -> Optional[Path]:
        """Get directory containing sample documents."""
        dataset_dir = self.datasets_dir / dataset_name.replace("/", "_")
        if dataset_dir.exists():
            return dataset_dir
        return None

    def delete_dataset(self, dataset_name: str) -> dict:
        """Delete a downloaded dataset."""
        dataset_dir = self.datasets_dir / dataset_name.replace("/", "_")

        if not dataset_dir.exists():
            return {
                "status": "error",
                "error": f"Dataset not found: {dataset_name}",
            }

        try:
            shutil.rmtree(dataset_dir)
            return {
                "status": "success",
                "message": f"Deleted dataset: {dataset_name}",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


# Global instance
_kaggle_instance: Optional[KaggleIntegration] = None


def get_kaggle() -> KaggleIntegration:
    """Get or create global Kaggle integration instance."""
    global _kaggle_instance
    if _kaggle_instance is None:
        _kaggle_instance = KaggleIntegration()
    return _kaggle_instance
