#!/usr/bin/env bash
# Install build dependencies for `make operators`
# Copyright Axelera AI, 2026

sudo apt-get update
sudo apt-get install -y \
    clinfo \
    g++ \
    gcc \
    glslang-tools \
    graphviz \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-rtsp \
    gstreamer1.0-tools \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libcairo2-dev \
    libeigen3-dev \
    libgirepository1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libavdevice-dev \
    libopencv-dev \
    libsimde-dev \
    libswscale-dev \
    libva-dev \
    libvulkan1 \
    ninja-build \
    nlohmann-json3-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    pkg-config \
    python3-dev \
    unzip \
    wget
