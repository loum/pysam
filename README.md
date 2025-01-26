# PySAM: Segment Anything Model 2 (SAM 2) using Python Ray

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Getting started](#getting-started)
  - [(macOS users only) upgrading GNU make](#macos-users-only-upgrading-gnu-make)
  - [Creating the local environment](#creating-the-local-environment)
- [Help](#help)
- [Running the `pysam` CLI](#running-the-pysam-cli)
  - [Ultralytics interface](#ultralytics-interface)
  - [Facebook Research interface](#facebook-research-interface)

## Overview

Create a Segment Anything Model 2 (SAM 2) compute pipeling using Python Ray, `pysam`

In the world of machine learning and computer vision, the process of making sense out of visual data is called 'inference' or 'prediction'. SAM 2 produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image.

Validated against Python versions [3.11.x](https://docs.python.org/3.11/).

## Prerequisites

- [GNU make](https://www.gnu.org/software/make/manual/make.html)
- Python 3 Interpreter. [We recommend installing pyenv](https://github.com/pyenv/pyenv)
- [Docker](https://www.docker.com/)
- [Makester project](https://github.com/loum/makester.git)

## Getting started

[Makester](https://loum.github.io/makester/) is used as the Integrated Developer Platform.

### (macOS users only) upgrading GNU make

Follow [these notes](https://loum.github.io/makester/macos/#upgrading-gnu-make-macos) to get
[GNU make](https://www.gnu.org/software/make/manual/make.html).

### Creating the local environment

Get the code and change into the top level `git` project directory:

```sh
git clone https://github.com/loum/pysam.git && cd pysam
```

> [!NOTE]
>
> Run all commands from the top-level directory of the `git` repository.

## Help

There should be a `make` target to be able to get most things done. Check the help for more information:

```
make help
```

## Running the `pysam` CLI

`pysam` allows you to generate SAM 2 segments on the CLI with the `pysam` tool. There are two interfaces that are currently supported

### Facebook Research interface (local)

The Facebook Research interface is based on the [segment-anything-2](https://github.com/facebookresearch/segment-anything-2) package and provide a richer experience around masking predictions.

```sh
pysam fbr predict --help
```

```sh
 Usage: pysam fbr predict [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --input-path            TEXT                               Source resource to feed into the Facebook Research SAM 2 predictor.        │
│                                                               [required]                                                                 │
│    --model                 [hiera_b|hiera_l|hiera_s|hiera_t]  The model pre-trained weights to use for predictions.                      │
│    --output-path           TEXT                               Directory to write out SAM 2 masks. [default: None]                        │
│    --flatten-output                                           Coalesce all mask output files to output path (ignore nested folders).     │
│    --output-file-format    [PNG|JPEG]                         The image file format to write with. [default: PNG]                        │
│    --help                                                     Show this message and exit.                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

The following example uses the test file set against the smallest of the SAM 2 pre-trained weights:

```sh
pysam fbr predict --model hiera_t --input-path tests/data/resources/images/png --output-path /tmp/images --flatten-output
```

> [!NOTE]
>
> Hiera Small, Hiera B+ and Hiera Large pre-trained weights are also supported, but are much larger in size. These are cached locally and only need to be downloaded once. Hiera Large is ~850 MB in size.

[top](#pysam-segment-anything-model-2-sam-2-using-python-ray)
