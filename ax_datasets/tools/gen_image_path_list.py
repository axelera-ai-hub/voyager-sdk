from pathlib import Path
import sys


# Function to list all image files with a specific format
def list_images_to_a_file(directory, output_file):
    # Get the directory where the output file will be saved
    output_dir = output_file.parent
    with output_file.open('w') as f:
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                # Write the relative path from the output file directory to the output file
                relative_path = './' + str(file_path.relative_to(output_dir))
                f.write(relative_path + '\n')


def main():
    if len(sys.argv) != 3:
        print("Usage: python gen_image_path_list.py <target/dir> <output.txt>")
        sys.exit(1)

    # Get the target directory and output file from the command line arguments
    target_dir = Path(sys.argv[1]).resolve()
    output_file = Path(sys.argv[2]).resolve()

    list_images_to_a_file(target_dir, output_file)


if __name__ == '__main__':
    main()
