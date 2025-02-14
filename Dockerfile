# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=gotman_1982/,target=gotman_1982/ \
    python -m pip install ./gotman_1982

# Switch to the non-privileged user to run the application.
USER appuser

# Create a data volume
VOLUME ["/data"]
VOLUME ["/output"]

# Define environment variables
ENV INPUT=""
ENV OUTPUT=""

# Run the application
CMD python irregulars_neureka_codebase/ioChallenge.py --pathIn "/data/$INPUT" --pathOut "/output/$OUTPUT"