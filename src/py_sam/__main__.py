"""Python Segment Anything Model 2 (pysam)."""

from dataclasses import dataclass
from enum import Enum
from typing import Type

import ray
import typer
from rich.console import Console
from rich.table import Table

import py_sam.model.context
from py_sam import fbr_sam
from py_sam.model import Model


@dataclass(frozen=True)
class FileFormatEnum(str, Enum):
    """ray.data.Dataset.write_images supported file formats."""

    PNG = "PNG"
    JPEG = "JPEG"


app = typer.Typer(
    add_completion=False, help="Python Segment-Anything Model CLI toolkit."
)

fbr_app = typer.Typer(add_completion=False, help="Facebook Research SAM 2 interface.")
app.add_typer(fbr_app, name="fbr")


@fbr_app.command("models")
def models(
    download: py_sam.model.context.FbrSamEnum = typer.Option(  # noqa: B008
        None,
        help="Download Facebook Research pre-trained weight.",
        show_choices=True,
        show_default=False,
    ),
    weights_list: bool = typer.Option(
        False, "--list", help="List local Facebook Research SAM 2 weights."
    ),
) -> None:
    """Facebook Research SAM 2 actions."""
    console = Console()

    if download == "hiera_b":
        py_sam.model.context.FbrSam.HIERA_B.value().download()
    elif download == "hiera_l":
        py_sam.model.context.FbrSam.HIERA_L.value().download()
    elif download == "hiera_s":
        py_sam.model.context.FbrSam.HIERA_S.value().download()
    elif download == "hiera_t":
        py_sam.model.context.FbrSam.HIERA_T.value().download()
    weights_list = True

    if weights_list:
        table = Table(title="Facebook Research SAM 2 weights in local cache")
        table.add_column("Path", justify="right", style="cyan", no_wrap=True)

        for weight_path in Model.get_weights():
            table.add_row(str(weight_path), style="yellow")

        console.print(table)


@fbr_app.command("predict")
def fbr_predict(
    input_path: str = typer.Option(
        help="Source resource to feed into the Facebook Research SAM 2 predictor.",
        show_default=False,
    ),
    model_type: py_sam.model.context.FbrSamEnum = typer.Option(  # noqa: B008
        None,
        "--model",
        help="The model pre-trained weights to use for predictions.",
        show_choices=True,
        show_default=False,
    ),
    output_path: str = typer.Option(None, help="Directory to write out SAM 2 masks."),
    flatten_output: bool = typer.Option(
        False,
        "--flatten-output",
        help="Coalesce all mask output files to output path (ignore nested folders).",
    ),
    output_file_format: FileFormatEnum = typer.Option(  # noqa: B008
        FileFormatEnum.PNG.value,
        "--output-file-format",
        help="The image file format to write with.",
        show_choices=True,
        show_default=True,
    ),
) -> None:
    """Facebook Research SAM 2 predict."""
    console = Console()

    model: Type[py_sam.model.Model] = py_sam.model.context.FbrSam.HIERA_L.value
    if model_type == "hiera_t":
        model = py_sam.model.context.FbrSam.HIERA_T.value
    elif model_type == "hiera_s":
        model = py_sam.model.context.FbrSam.HIERA_S.value
    elif model_type == "hiera_b":
        model = py_sam.model.context.FbrSam.HIERA_B.value

    console.print(f"📐 Model pre-trained weight: {model}")

    ray.init()
    fbr_sam.FbrSam.process(
        source_data_path=input_path,
        model=model,
        output_path=output_path,
        flatten_output=flatten_output,
        file_format=output_file_format,
    )


def main() -> None:
    """Script entry point."""
    app()


if __name__ == "__main__":
    main()
