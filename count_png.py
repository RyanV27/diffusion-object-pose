import os

def count_png_files(directory):
    png_count = 0
    total_png_count = 0
    for root, _, files in os.walk(directory):
        if(os.path.basename(root) == ".ipynb_checkpoints"):
            continue
        png_count = sum(1 for file in files if file.lower().endswith('.png') and file.lower().startswith('m'))
        total_png_count += png_count
        print(f"{os.path.basename(root)} : {png_count}")
    return total_png_count

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    directory = os.path.abspath(directory)
    print("Checking for directory: ", directory)
    png_files = count_png_files(directory)
    print(f"Total .png files found: {png_files}")