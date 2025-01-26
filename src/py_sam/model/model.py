"""Base abstract class for pre-trained weight SAM 2 models."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator

import httpx
import urllib3.exceptions
from huggingface_hub import hf_hub_download
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from py_sam.logging_config import log


class Model(ABC):
    """Abstraction of the Segment Anything Model 2 (SAM 2) model facility."""

    def __init__(self) -> None:
        """Initialise the `py_sam.model.Model` class."""
        self.__target_basename: Path = Path(
            os.environ.get("MODEL_CACHE", Path.home() / ".cache" / "blade" / "models")
        )

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Getter for the model type."""

    @property
    def target_basename(self) -> Path:
        """Getter for the `target_basename` attribute."""
        return self.__target_basename

    @property
    @abstractmethod
    def repo_id(self) -> str:
        """Getter for the `repo_id` attribute."""

    @property
    @abstractmethod
    def filename(self) -> str:
        """Getter for the `filename` attribute."""

    @property
    @abstractmethod
    def model_cfg(self) -> str:
        """Getter for the `model_cfg` attribute."""

    @property
    def checkpoint(self) -> Path:
        """Getter for the model name attribute."""
        return self.target_basename / self.filename

    @retry(
        retry=(
            retry_if_exception_type(httpx.HTTPError)
            | retry_if_exception_type(urllib3.exceptions.HTTPError)
        ),
        stop=stop_after_attempt(10),
        wait=wait_fixed(60),
    )
    def download(self) -> None:
        """Download a SAM 2 model if not already cached locally.

        Returns
            Fully qualified name of target path on download success. `None` otherwise.

        """
        if self.checkpoint.exists() and self.checkpoint.is_file():
            log.info(f'Model "{self.checkpoint}" exists. Skipping download.')
            return

        self.target_basename.mkdir(parents=True, exist_ok=True)

        log.info(
            f"Downloading model {self.filename} from {self.repo_id} to {self.target_basename}"
        )
        hf_hub_download(
            repo_id=self.repo_id, filename=self.filename, local_dir=self.target_basename
        )

        if self.checkpoint.exists() and self.checkpoint.is_file():
            log.info(f"Model cache location: {self.checkpoint}")
        else:
            log.error(f"Model download error. {self.checkpoint} does not exist.")

    @staticmethod
    def get_weights() -> Generator[Path, None, None]:
        """List the local cached SAM 2 weights."""
        return Path(
            os.environ.get("MODEL_CACHE", Path.home() / ".cache" / "blade" / "models")
        ).glob("*.pt")
