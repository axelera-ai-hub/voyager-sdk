#!/usr/bin/env python
# Copyright Axelera AI, 2026

import json
import os
import subprocess
import sys
import re
from sys import platform

# Suppress OpenCV warnings - must be set before importing cv2
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2  # noqa: E402
import contextlib  # noqa: E402
from typing import Any  # noqa: E402


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr and stdout output at the file descriptor level.
    Doing the necessary tests to discover all the connected cameras, will inevitably throw up
    warning and error. This function, used with a `with` operator, will suppress those messages.
    Works on both Windows and Unix-like systems.
    """
    # Save the original stderr and stdout file descriptors
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    old_stderr_fd = os.dup(stderr_fd)
    old_stdout_fd = os.dup(stdout_fd)

    # Open devnull (works on both Windows and Unix)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        # Redirect both stderr and stdout to devnull at the file descriptor level
        os.dup2(devnull_fd, stderr_fd)
        os.dup2(devnull_fd, stdout_fd)
        # Flush to ensure all buffered output is cleared
        sys.stderr.flush()
        sys.stdout.flush()
        yield
    finally:
        # Flush before restoring
        sys.stderr.flush()
        sys.stdout.flush()
        # Restore stderr and stdout
        os.dup2(old_stderr_fd, stderr_fd)
        os.dup2(old_stdout_fd, stdout_fd)
        os.close(old_stderr_fd)
        os.close(old_stdout_fd)
        os.close(devnull_fd)


# Common resolutions to test
resolutions_to_test_for = [
    (320, 240),  # QVGA
    (640, 480),  # VGA
    (800, 600),  # SVGA
    (960, 544),  # custom
    (960, 720),  # HD-ready
    (1024, 768),  # XGA
    (1280, 720),  # HD 720p
    (1280, 1024),  # SXGA
    (1600, 1200),  # UXGA
    (1920, 1080),  # Full HD 1080p
    (2560, 1440),  # QHD
    (3840, 2160),  # 4K UHD
    (4096, 2160),  # 4K DCI
    (7680, 4320),  # 8K UHD
    (8192, 4320),  # 8K DCI
]


# Common FOURCC codes to test
formats_to_test_for = [
    ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
    ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
    ('H264', cv2.VideoWriter_fourcc(*'H264')),
    ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
    ('MPEG', cv2.VideoWriter_fourcc(*'MPEG')),
    ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),
    ('YUY2', cv2.VideoWriter_fourcc(*'YUY2')),
    ('GREY', cv2.VideoWriter_fourcc(*'GREY')),
]


framerates_to_test_for = [5, 10, 15, 20, 24, 25, 30, 50, 60, 90, 120, 240]


def _fourcc_to_string(fourcc: int) -> str:
    """
    Convert FOURCC code to readable string.

    Args:
        fourcc (int): FOURCC code as integer

    Returns:
        str: FOURCC code as string
    """
    if fourcc == -1 or fourcc == 0:
        return "Unknown"

    return "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])


def _get_framerates(
    cap: cv2.VideoCapture, width: int, height: int, fourcc_code: int
) -> list[float]:
    """
    Test which framerates are supported for a given resolution and format.

    Args:
        cap: OpenCV VideoCapture object
        width (int): Frame width
        height (int): Frame height
        fourcc_code: FOURCC code for the format

    Returns:
        list: List of supported FPS values
    """
    supported_fps = []
    for fps in framerates_to_test_for:
        cap.set(cv2.CAP_PROP_FPS, fps)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Consider it supported if the actual FPS is close to requested
        if abs(actual_fps - fps) < 1.0 and actual_fps not in supported_fps:
            supported_fps.append(actual_fps)

    # If no specific framerates were set, just return the current one
    if not supported_fps:
        current_fps = cap.get(cv2.CAP_PROP_FPS)
        if current_fps > 0:
            supported_fps.append(current_fps)

    return sorted(set(supported_fps))


def get_camera_properties(camera_index: int) -> dict[str, Any]:
    """
    Interrogate a USB camera to get its supported properties.

    Args:
        camera_index (int): Camera device index (default: 0)

    Returns:
        dict: Dictionary containing camera properties including supported resolutions,
              framerates, and formats
    """
    camera_info = {}

    with suppress_stderr():
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            return {}

        camera_info = {
            'camera_index': camera_index,
            'backend': cap.getBackendName(),
            'supported_configs': [],
        }

        # Get current/default properties
        default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        default_fps = cap.get(cv2.CAP_PROP_FPS)
        default_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        default_format = _fourcc_to_string(default_fourcc)

        camera_info['default_config'] = {
            'resolution': (int(default_width), int(default_height)),
            'fps': default_fps,
            'format': default_format,
            'fourcc_code': default_fourcc,
        }

        # Test each format with each resolution
        for format_name, fourcc_code in formats_to_test_for:
            for width, height in resolutions_to_test_for:
                # Try setting the format
                cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

                # Try setting resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                # Read back what was actually set
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                actual_format = _fourcc_to_string(actual_fourcc)

                # Check if the camera accepted our settings
                if (
                    actual_width == width
                    and actual_height == height
                    and actual_format == format_name
                ):
                    # Test different framerates
                    fps_values = _get_framerates(cap, width, height, fourcc_code)

                    config = {
                        'resolution': (width, height),
                        'format': format_name,
                        'fourcc_code': actual_fourcc,
                        'supported_fps': fps_values,
                    }

                    # Avoid duplicates
                    if config not in camera_info['supported_configs']:
                        camera_info['supported_configs'].append(config)

        cap.release()
    return camera_info


def _get_windows_camera_indices(max_cameras: int) -> list[int]:
    """
    Query Windows PnP devices via PowerShell to determine how many cameras are present,
    returning a range limited to that count rather than blindly iterating up to max_cameras.
    Falls back to range(max_cameras) if the query fails.
    """
    try:
        result = subprocess.run(
            [
                'powershell',
                '-NoProfile',
                '-Command',
                '(Get-PnpDevice -Class Camera -Status OK).Count',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip().isdigit():
            return list(range(int(result.stdout.strip())))
    except Exception:
        pass
    return list(range(max_cameras))


def _get_macos_camera_indices(max_cameras: int) -> list[int]:
    """
    Query macOS system_profiler to determine how many cameras are present,
    returning a range limited to that count rather than blindly iterating up to max_cameras.
    Falls back to range(max_cameras) if the query fails.
    """
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType', '-json'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            cameras = data.get('SPCameraDataType', [])
            return list(range(len(cameras)))
    except Exception:
        pass
    return list(range(max_cameras))


def scan_all_cameras(max_cameras: int) -> list[dict[str, Any]]:
    """
    Scan from 0 to max_cameras-1 for cameras present.
    if present get the properties of the camera and add to list of available cameras.

    Args:
        max_cameras (int): Maximum number of camera indices to check

    Returns:
        list[dict[str, Any]]: List of camera info dictionaries
    """
    camera_indexes: list[int] = []
    using_max_camera_range = False
    if platform in ("linux", "linux2"):
        # Linux exposes cameras as /dev/videoN character devices.
        camera_indexes = sorted(
            [int(x.removeprefix('video')) for x in os.listdir('/dev') if re.match(r'video\d+$', x)]
        )
    elif platform == "darwin":
        # macOS does not use /dev/videoN; query system_profiler instead.
        camera_indexes = _get_macos_camera_indices(max_cameras)
        using_max_camera_range = bool(len(camera_indexes) == max_cameras)
    else:
        # Ask Windows which cameras are present rather than iterating a large blind range.
        # Falls back to range(max_cameras) with a sequential break if the query fails.
        camera_indexes = _get_windows_camera_indices(max_cameras)
        using_max_camera_range = bool(len(camera_indexes) == max_cameras)

    available_cameras: list[dict[str, Any]] = []
    for i in camera_indexes:
        camera_info = get_camera_properties(i)
        if camera_info:
            available_cameras.append(camera_info)
        elif using_max_camera_range and platform not in ("linux", "linux2"):
            break
    return available_cameras


def main() -> None:
    """Prints the properties of all available cameras to the console in a human-readable format."""

    def _resolution_tuple_to_str(res: tuple[int, int]) -> str:
        return f"{res[0]:>4}x{res[1]:>4}"

    import time

    print("Scanning for cameras...")
    start_time = time.time()
    available_cameras = scan_all_cameras(100)

    for cam in available_cameras:
        print("")
        print(f"Camera Index: {cam['camera_index']}")
        print(f"Backend: {cam['backend']}")
        print(
            "Default Config: Resolution: "
            f"{_resolution_tuple_to_str(cam['default_config']['resolution'])}, "
            f"Format: {cam['default_config']['format']}, FPS: {cam['default_config']['fps']}"
        )
        print("Supported Configurations:")
        for config in cam['supported_configs']:
            print(
                f"    Resolution: {_resolution_tuple_to_str(config['resolution'])}, "
                f"Format: {config['format']}, Supported FPS: {config['supported_fps']}"
            )
    print(f"\nCamera scan completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
