#! /usr/bin/env python3
# List all image files in a directory and save to a text file
# Usage:
#   tools/list_relative_image_paths.py ./data/dataset/val.txt ./data/dataset/valid/images/
import argparse
import os
import sys


def list_image_files(images_dir, output_file):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    image_files = []

    output_dir = os.path.dirname(os.path.abspath(output_file))

    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_dir)
                image_files.append(f"./{rel_path}")

    return image_files


def main():
    parser = argparse.ArgumentParser(
        description="List image files in a directory and save relative paths to a text file."
    )
    parser.add_argument("output_file", help="Path to the output text file")
    parser.add_argument("images_dir", help="Path to the directory containing images")
    args = parser.parse_args()

    if not os.path.isdir(args.images_dir):
        print(f"Error: '{args.images_dir}' is not a valid directory.")
        sys.exit(1)

    image_list = list_image_files(args.images_dir, args.output_file)

    try:
        with open(args.output_file, 'w') as f:
            for image_path in image_list:
                f.write(f"{image_path}\n")
        print(f"Image list with relative paths has been saved to: {args.output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

    print(args.output_file)


if __name__ == "__main__":
    main()
