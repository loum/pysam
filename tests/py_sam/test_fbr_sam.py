"""Segment Anything Model 2 (SAM 2) unit tests."""

from pathlib import Path

import numpy as np
import pytest
import ray
from py_sam.fbr_sam import FbrSam, ImageFilenameProvider


def test_fbr_sam_init() -> None:
    """Initialise a FbrSam object."""
    # Given an initialised a FbrSam
    sam = FbrSam()

    # I should get a FbrSam instance
    assert isinstance(sam, FbrSam), "Object is not a FbrSam instance"


@pytest.mark.skipif(
    not (Path.home() / ".cache" / "py-sam" / "models" / "sam2_hiera_large.pt").exists(),
    reason="Unable to find SAM 2 pre-trained weights.",
)
def test_generate_masks(data_dir: Path) -> None:
    """Generate the SAM 2 prediction masks."""
    # Given an image
    source_image = data_dir / "png" / "cat.png"

    sam = FbrSam()
    sam.generate_masks(image=source_image, image_name=source_image.name)


FILENAME_FORMAT_ARGS: tuple = (
    "PNG",
    "JPEG",
)


@pytest.mark.skipif(
    not (Path.home() / ".cache" / "py-sam" / "models" / "sam2_hiera_large.pt").exists(),
    reason="Unable to find SAM 2 pre-trained weights.",
)
@pytest.mark.parametrize("filename_format_args", FILENAME_FORMAT_ARGS)
def test_ray_process(
    data_dir: Path, tmp_path: Path, filename_format_args: str, ray_session: None
) -> None:
    """Ray SAM 2 batch processing."""
    # Given a source directory path
    source_data_path = str(data_dir / filename_format_args.lower())

    # when I batch process the SAM 2 mask generation
    FbrSam.process(
        source_data_path=source_data_path,
        output_path=f"file://{tmp_path}",
        flatten_output=True,
        file_format=filename_format_args,
    )


def test_image_convert_from_ray_ingest(data_dir: Path) -> None:
    """Standardise the image format for further processing - ray ingest."""
    # Given a Ray Dataset source
    dset = ray.data.read_images(data_dir, include_paths=True)

    # when I attempt to standardise the input image sources as Numpy arrays
    converted_images = [
        FbrSam.image_convert(image=row.get("image")) for row in dset.iter_rows()
    ]

    # then the images should remain as Numpy arrays
    assert all(
        isinstance(x, np.ndarray) for x in converted_images
    ), "Converted images are not all Numpy arrays."


def test_image_convert_from_image_path_ingest(data_dir: Path) -> None:
    """Standardise the image format for further processing - image path ingest."""
    # Given a source image path
    source_image = data_dir / "png" / "cat.png"

    # when I attempt to standardise the input image sources as Numpy arrays
    converted_image = FbrSam.image_convert(image=source_image)

    # then the images should be converted into a Numpy array
    assert isinstance(
        converted_image, np.ndarray
    ), "Converted images not a Numpy array."


def test_image_filename_provider_init() -> None:
    """Initialise a ImageFilenameProvider object."""
    # Given an initialised a FbrSam
    filename_provider = ImageFilenameProvider()

    # I should get a FbrSam instance
    assert isinstance(
        filename_provider, ImageFilenameProvider
    ), "Object is not a ImageFilenameProvider instance"


FILENAME_SPLITTER_ARGS: tuple = (
    (
        {
            "filename": "s3://tester/images",
        },
        ("/images", "images", []),
    ),
    (
        {
            "filename": "s3://tester/images/cat.png",
        },
        ("/images/", "cat", [".png"]),
    ),
    (
        {
            "filename": "s3://tester/images/abc/myimage.png",
        },
        ("/images/abc/", "myimage", [".png"]),
    ),
    (
        {
            "filename": "tester/images/abc/myimage.png",
        },
        ("images/abc/", "myimage", [".png"]),
    ),
    (
        {
            "filename": "tester/myimage.png",
        },
        ("", "myimage", [".png"]),
    ),
)


@pytest.mark.parametrize(
    "filename_splitter_kwargs,filename_splitter_expected", FILENAME_SPLITTER_ARGS
)
def test_filename_splitter(
    filename_splitter_kwargs: dict,
    filename_splitter_expected: tuple[str, str, list],
) -> None:
    """Filename splitter across the dot ('.') character."""
    # Given a fully qualified filename
    # filename_splitter_kwargs

    # when I split the filename components
    filename_parts = ImageFilenameProvider.filename_splitter(**filename_splitter_kwargs)

    # then I should receive the individual file components.
    assert filename_parts == filename_splitter_expected
