# syntax=docker/dockerfile:1.4

FROM python:3.11-slim-buster

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# For local PyPI artifact builds.
ARG BUILT_DISTRIBUTION_NAME
COPY dist/$BUILT_DISTRIBUTION_NAME dist/

# Install Ray to act as the RayJob submitter.
# Remove redundant artifacts from site-packages to reduce container image size.
RUN python -m pip install --no-cache-dir --upgrade pip &&\
 python -m pip install --no-compile --no-cache-dir --no-warn-script-location\
 dist/$BUILT_DISTRIBUTION_NAME\
 ray[default]~=2.34 &&\
 find /usr/local/lib/python3.11/site-packages/ -depth\
   \(\
    \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
     -o \
     \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
   \) -exec rm -rf '{}' +;

WORKDIR /opt/py-sam/app

COPY docker/scripts/bootstrap.sh /bootstrap.sh

ARG RAY_ADDRESS
ENV RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}

ARG PY_SAM__NUM_GPUS
ENV PY_SAM__NUM_GPUS=${PY_SAM__NUM_GPUS}
ARG PY_SAM__NUM_CPUS
ENV PY_SAM__NUM_CPUS=${PY_SAM__NUM_CPUS}
ARG PY_SAM__CONCURRENCY
ENV PY_SAM__CONCURRENCY=${PY_SAM__CONCURRENCY}

ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ARG MINIO_URL
ENV MINIO_URL=${MINIO_URL}
ARG MINIO_SSL_VERIFY=true
ENV MINIO_SSL_VERIFY=${MINIO_SSL_VERIFY}

ENTRYPOINT ["/bootstrap.sh"]
CMD []
