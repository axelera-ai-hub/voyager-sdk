#!/bin/bash
# Copyright Axelera AI, 2026

axdownloadmodel yolov5s-v7-coco --version 1.8.0-a0 --force || true
./inference.py yolov5s-v7-coco media/traffic3_480p.mp4 --no-display || true
axrunmodel build/yolov5s-v7-coco/yolov5s-v7-coco/1/model.json || true
