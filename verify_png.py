import os
import re
import argparse
from PIL import Image

# Command to run in PR_Project/diff-feats-pose: python verify_png.py --dir ./dataset/crop_image512/LINEMOD/{object_name}
def check_png_files(directory):
    # Loop through the directory and its subdirectories
    if not os.path.exists(directory):
        print(f"{directory} does not exist!")
        return
        
    for dirpath, dirnames, filenames in os.walk(directory):
        print(f"In subdirectory: {dirpath}")
        for filename in filenames:
            if filename.lower().endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                try:
                    # Try to open the .png file
                    with Image.open(file_path) as img:
                        img.verify()  # Verify if the image is corrupted
                    # print(f"{file_path}: OK")
                except (IOError, SyntaxError) as e:
                    print(f"{file_path}: Corrupted ({e})")
    
def findMissing(directory):
    if not os.path.exists(directory):
        print(f"{directory} does not exist!")
        return
        
    pattern = re.compile(r'^\d{6}\.png$')
    for root, dir_names, file_names in os.walk(directory):
        print(f"In directory: {root}")
        for file in file_names:
            if(pattern.match(file) and not (os.path.exists(root + '/' + file[:-4] + '_mask.png'))):
                print(f"Mask missing for {file}\n{root + '/' + file[:-4] + '_mask.png'}")
                # os.remove(root + '/' + file)
                # print(f"Removed {root + '/' + file}")

if __name__ == "__main__":
    # directory = input("Enter the directory path: ")
    # directory = os.path.abspath(directory)

     # Set up argument parsing
    parser = argparse.ArgumentParser(description="Check if .png files in a directory are corrupted")
    parser.add_argument('--dir', type=str, help="Path to the directory to check")
    args = parser.parse_args()
    
    print("Checking for directory: ", args.dir)
    png_files = findMissing(args.dir)
    # check_png_files(args.dir)