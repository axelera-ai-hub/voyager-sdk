#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Installs a demo by creating a desktop file and a runfile.

import argparse
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import subprocess

from axelera.app import config

# Define path and get list of demo files
demos_path = os.path.join(config.env.framework, 'examples/demos')
files = [f for f in listdir(demos_path) if isfile(join(demos_path, f)) and f.endswith('demo.py')]

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Install a demo by creating a desktop file and a runfile.'
)
parser.add_argument(
    'demo',
    type=str,
    help='Name of the demo to install. You can choose from the following: '
    + ', '.join([f[:-8] for f in files]),
)
args = parser.parse_args()

if args.demo not in [f[:-8] for f in files]:
    print(
        f"Demo '{args.demo}' not found. Available demos are: " + ', '.join([f[:-8] for f in files])
    )
    exit(1)

home = Path.home()
if not os.path.exists(home):
    print("Cannot find home folder. Please check your environment.")
    exit(1)


def find_camera(max_index=25):
    import cv2

    # os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    cv2.setLogLevel(0)  # 0 = silent

    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()

    if len(available) == 0:
        print("Error: No cameras found")
        exit(0)
    elif len(available) > 1:
        print(f"Warning: more than 1 cameras found, using camera {available[0]}")

    return available[0]


# Define demo name and file names
demo_name = args.demo
desktop_file = f"{demo_name}.desktop"
runfile = f"run_{demo_name}.sh"

# Create runfile
extra = ""
if demo_name == "fruit":
    cam = find_camera()
    extra = f"usb:{cam} "
    print(extra)

with open(os.path.join(home, runfile), 'w') as f:
    f.write("#!/bin/bash\n")
    f.write(f"cd {config.env.framework}\n")
    f.write("source venv/bin/activate\n")
    f.write(f"./examples/demos/{demo_name}_demo.py {extra}--window-size=fullscreen\n")

# Make runfile executable
os.chmod(os.path.join(home, runfile), 0o777)

# Create desktop file
with open(os.path.join(home, 'Desktop', desktop_file), 'w') as f:
    f.write("[Desktop Entry]\n")
    f.write("Type=Application\n")
    f.write(f"Name={str(demo_name).title()} demo\n")
    f.write(f"Exec={os.path.join(home, runfile)}\n")
    f.write(
        f"Icon={os.path.join(config.env.framework, 'axelera/app/render_assets/axelera-ai-logo.png')}\n"
    )  # Placeholder icon
    f.write("Terminal=true\n")
    f.write("Categories=AxeleraAI;Demo;\n")

# Allow the desktop file to launch
subprocess.run(
    ['gio', 'set', os.path.join(home, 'Desktop', desktop_file), 'metadata::trusted', 'true'],
    check=True,
)

# Make desktop file executable
os.chmod(os.path.join(home, 'Desktop', desktop_file), 0o777)

# Print success message
print(
    f"Demo '{demo_name}' installed successfully! Some demos may require a source to be specified in the runscript. This script has attempted to find the right one. Please check the runscript {runfile} in your home directory."
)
