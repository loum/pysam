"""Facebook Research Segment Anything 2 Model (SAM 2) tooling."""

import os
from pathlib import Path
from typing import Type, cast
from urllib.parse import urlparse

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from dotenv import load_dotenv
from pyarrow.fs import FSSpecHandler, PyFileSystem  # type: ignore[import-untyped]
from ray.data.datasource import FilenameProvider
from s3fs import S3FileSystem  # type: ignore[import-untyped]
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # type: ignore[import-untyped]
from sam2.build_sam import build_sam2  # type: ignore[import-untyped]

import py_sam.model.hiera
from py_sam.logging_config import log

load_dotenv()
matplotlib.use("agg")


class ImageFilenameProvider(FilenameProvider):
    """Customised filename provider."""

    def __init__(self, mask_symbol: str = "masks") -> None:
        """Initialise a ImageFilenameProvider instance."""
        self.__mask_symbol = mask_symbol

    @property
    def mask_symbol(self) -> str:
        """Mask token getter."""
        return self.__mask_symbol

    def get_filename_for_row(  # type: ignore[no-untyped-def]
        self, row, task_index, block_index, row_index
    ) -> str:
        """Customised filename generator.

        Parameters:
            row:
            task_index:
            block_index:
            row_index:

        Returns:
            The new masked filename.

        """
        filename_base, filename_stem, filename_suffixes = (
            ImageFilenameProvider.filename_splitter(row["path"])
        )

        masked_output = f"{filename_base or ''}{filename_stem}_{self.mask_symbol}{''.join(filename_suffixes)}"
        if row["flatten_output"]:
            masked_output = (
                f"{filename_stem}_{self.mask_symbol}{''.join(filename_suffixes)}"
            )

        return masked_output

    @staticmethod
    def filename_splitter(filename: str) -> tuple[str | None, str, list]:
        """Filename splitter across the dot ('.') character.

        Returns:
            Tuple structure that represents the original filename construct.

        """
        filename_name = Path(filename.split("/")[-1])
        filename_parent = None
        uri_filename_parsed = urlparse(filename)
        if uri_filename_parsed.scheme in ["s3"]:
            filename_parent = uri_filename_parsed.path
        elif not uri_filename_parsed.scheme:
            # Special case where Ray read_images strips the scheme from the path.
            filename_parent = uri_filename_parsed.path.split("/", 1)[1]

        if filename_parent is not None and "." in str(filename_name):
            filename_parent = filename_parent.replace(str(filename_name), "")

        return filename_parent, filename_name.stem, filename_name.suffixes


class FbrSam:
    """Facebook Research Segment Anything Model (SAM) abstraction."""

    def __init__(
        self,
        device: str = "cpu",
        model: Type[py_sam.model.Model] = py_sam.model.hiera.HieraLarge,
    ) -> None:
        """Initialise a Facebook Research Segment Anything Model (SAM) instance."""
        if torch.backends.mps.is_available():
            self.__device = "mps"
        else:
            self.__device = "cuda" if torch.cuda.is_available() else device
        log.info(f"SAM device initialised as: {self.__device}")

        self.__model = model()
        log.info(f"SAM pre-trained weight initialised as: {self.__model.model_type}")

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Batch compute."""
        batch["image"] = self.generate_masks(
            image=batch["image"], image_name=str(batch["path"])
        )

        return batch

    @property
    def device(self) -> str:
        """SAM device getter."""
        return self.__device

    @property
    def model(self) -> py_sam.model.Model:
        """SAM model getter."""
        return self.__model

    @staticmethod
    def image_convert(image: str | Path | np.ndarray) -> np.ndarray:
        """Standardise the image format for further processing.

        `image` can either be a path to the original image source or a `numpy` array structure.

        Parameters:
            image: The source image to standardise.

        """
        transformed_image = (
            cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
        )

        return cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

    def image_spec(
        self,
        image: np.ndarray,
        image_name: str,
    ) -> tuple[int, int]:
        """Extract the image specifications.

        Parameters:
            image:
            image_name:

        Returns:
            The image height and width as integers in a tuple structure.

        """
        height, width, channels = image.shape
        log.info(
            f"Image spec for {image_name} - "
            f"height: {height} | width: {width} | channels: {channels} | size: {image.size} bytes"
        )

        return height, width

    def generate_masks(
        self, image: str | Path | np.ndarray, image_name: str
    ) -> np.ndarray:
        """Generate the SAM prediction masks.

        Parameters:
            image:
            image_name:

        """
        image = self.image_convert(image)
        orig_height, orig_width = self.image_spec(image, image_name)

        self.model.download()

        sam = build_sam2(
            self.model.model_cfg,
            self.model.checkpoint,
            device=self.device,
            apply_postprocessing=False,
        )

        mask_generator = SAM2AutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)

        px = 1 / float(plt.rcParams.get("figure.dpi", 100.0))
        plt.figure(figsize=(orig_width * px, orig_height * px))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(image)
        self.show_annotations(masks)

        canvas = cast(plt.Figure, plt.gca().figure).canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)  # type: ignore[attr-defined]

        return data.reshape(canvas.get_width_height()[::-1] + (3,))

    @staticmethod
    def show_annotations(anns: dict) -> None:
        """Apply the masks via the set of image annotations.

        Parameters:
            anns: The image annotations.

        """
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

    @staticmethod
    def process(
        source_data_path: str,
        model: Type[py_sam.model.Model] = py_sam.model.hiera.HieraLarge,
        output_path: str | None = None,
        flatten_output: bool = False,
        file_format: str = "PNG",
    ) -> None:
        """Ray SAM batch processing.

        source_data_path: Location of the source images to run the SAM predictions over.
        model: The pre-trained weight to use for the compute.
        output_path: Override the path to write the masks to.
        flatten_output: Only use filename as output to `output_path`.
        file_format: The image file format to write with.

        """
        log.info(f"Source data path: {source_data_path}")

        uri_parsed = urlparse(source_data_path)

        pa_fs = None
        if uri_parsed.scheme in ["s3"]:
            filesystem = S3FileSystem(
                key=os.environ.get("AWS_ACCESS_KEY_ID"),
                secret=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                client_kwargs={
                    "endpoint_url": os.environ.get("MINIO_URL"),
                    "verify": os.environ.get("MINIO_SSL_VERIFY", "true") == "true",
                },
            )
            pa_fs = PyFileSystem(FSSpecHandler(filesystem))

        num_cpus = os.environ.get("PY_SAM__NUM_CPUS")
        num_gpus = os.environ.get("PY_SAM__NUM_GPUS")
        log.info(f"Overriding values for CPU/GPU: {num_cpus}/{num_gpus}")

        ray.data.read_images(
            paths=source_data_path,
            filesystem=pa_fs,
            include_paths=True,
            concurrency=int(os.environ.get("PY_SAM__CONCURRENCY", 1)),
        ).add_column("flatten_output", lambda df: flatten_output).map(
            FbrSam,
            fn_constructor_kwargs={"model": model},
            num_cpus=int(num_cpus) if num_cpus else None,
            num_gpus=int(num_gpus) if num_gpus else None,
            concurrency=int(os.environ.get("PY_SAM__CONCURRENCY", 1)),
        ).write_images(
            path=output_path,
            filesystem=pa_fs,
            column="image",
            file_format=file_format,
            concurrency=int(os.environ.get("PY_SAM__CONCURRENCY", 1)),
            filename_provider=ImageFilenameProvider(),
        )
