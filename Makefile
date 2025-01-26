.SILENT:
.DEFAULT_GOAL := help

# Use a single bash shell for each job, and immediately exit on failure
ifeq (,$(shell which zsh))
SHELL := bash
else
SHELL := zsh
endif
.SHELLFLAGS := -ceu
.ONESHELL:

#
# Makester overrides.
#
MAKESTER__STANDALONE := true
MAKESTER__INCLUDES := py docker versioning
MAKESTER__REPO_NAME := loum

include $(HOME)/.makester/makefiles/makester.mk

MAKESTER__PROJECT_NAME := py-sam
MAKESTER__VERSION_FILE := src/$(MAKESTER__PACKAGE_NAME)/VERSION
MAKESTER__VENV_HOME := $(MAKESTER__PROJECT_DIR)/.venv

# Container image build.
#
# Image versioning follows the format: <jupyter-version>-<spark-version>-<image-release-number>
#
MAKESTER__VERSION := $(JUPYTER_VERSION)-$(SPARK_VERSION)
MAKESTER__RELEASE_NUMBER := 1

SPARK_BASE_IMAGE := loum/pyjdk:python3.11-openjdk11
JUPYTER_PORT ?= 8889
MAKESTER__BUILD_COMMAND := --rm --no-cache\
 --tag $(MAKESTER__IMAGE_TAG_ALIAS)\
 --file docker/Dockerfile .

#
# Local Makefile targets.
#

# Build the local development environment.
#
init-dev: py-venv-clear py-venv-init
	MAKESTER__PIP_INSTALL_EXTRAS=dev $(MAKE) py-install-extras

# Streamlined production packages.
#
init: _venv-init
	$(MAKE) py-install

# pysam test harness.
#
ifeq ($(TESTS_TO_RUN),tests)
  _COVERAGE := --cov src
endif

tests:
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(MAKESTER__PYTHON) -m pytest $(_COVERAGE) $(TESTS_TO_RUN)

help: makester-help
	printf "\n(Makefile)\n"
	$(call help-line,init,Build \"py-sam\" environment streamlined for production releases)
	$(call help-line,init-dev,Build \"py-sam\" environment)
	$(call help-line,tests,Run the test suite)

.PHONY: tests
