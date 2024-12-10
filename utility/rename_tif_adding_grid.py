#!/usr/bin/env python
# Filename: rename_tif_adding_grid.py 
"""
introduction: adding correct grid number to tif files

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 10 December, 2024
"""

import os, sys

from datetime import datetime

def are_files_identical(file1, file2):
    """
    Check if two files are identical by comparing their sizes and contents.

    Parameters:
    file1 (str): Path to the first file.
    file2 (str): Path to the second file.

    Returns:
    bool: True if the files are identical, False otherwise.
    """
    # First compare file sizes
    if os.path.getsize(file1) != os.path.getsize(file2):
        return False

    # Compare file contents byte-by-byte
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            chunk1 = f1.read(8192)  # Read in chunks of 8 KB
            chunk2 = f2.read(8192)
            if chunk1 != chunk2:  # Compare the current chunks
                return False
            if not chunk1:  # End of file reached
                break

    return True


def rename_tif_by_adding_correct_gridNum_one_region(region_dir, ref_dir):
    """
    Rename .tif files in subfolders of `region_dir` by matching their `_number.tif` suffix with files in `subImages` in `ref_dir`,
    only if the files are identical. Also, update the filenames in .txt files in subfolders of `region_dir`.

    Write by AI

    Parameters:
    region_dir (str): Directory containing subfolders with .tif files and .txt files.
    ref_dir (str): Directory containing the "subImages" folder with reference .tif files.
    """
    # Define the path to the "subImages" folder
    sub_images_dir = os.path.join(ref_dir, "subImages")
    if not os.path.exists(sub_images_dir):
        print(f"Error: The 'subImages' folder does not exist in {ref_dir}.")
        return

    # Create a mapping of `_number.tif` to full filenames from the subImages folder
    sub_images_files = {}
    for file in os.listdir(sub_images_dir):
        if file.endswith(".tif"):
            key = file.split("_")[-1]  # Extract `_number.tif`
            sub_images_files[key] = file  # Map `_number.tif` to full filename

    print(f"Loaded {len(sub_images_files)} reference .tif files from 'subImages'.")

    # Recursively process all subfolders in the region directory
    for root, _, files in os.walk(region_dir):
        for file in files:
            if file.endswith(".tif"):
                key = file.split("_")[-1]  # Extract `_number.tif`
                if key in sub_images_files:
                    # Paths for the files to compare
                    old_path = os.path.join(root, file)
                    ref_file = sub_images_files[key]
                    ref_path = os.path.join(sub_images_dir, ref_file)

                    # Verify if the files are identical
                    if not are_files_identical(old_path, ref_path):
                        print(f"Error: Files {file} and {ref_file} are not identical. Aborting.")
                        sys.exit(1)  # Exit the program

                    # Rename the .tif file
                    new_name = ref_file
                    new_path = os.path.join(root, new_name)
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {file} -> {new_name}")
                    except Exception as e:
                        print(f"Error renaming {file}: {e}")

                    # Update the filenames in .txt files in the same folder
                    for txt_file in files:
                        if txt_file.endswith(".txt"):
                            txt_path = os.path.join(root, txt_file)
                            try:
                                with open(txt_path, "r") as f:
                                    content = f.read()

                                # Replace old filename with new filename
                                updated_content = content.replace(file, new_name)

                                with open(txt_path, "w") as f:
                                    f.write(updated_content)

                                print(f"Updated {txt_file}: {file} -> {new_name}")
                            except Exception as e:
                                print(f"Error updating {txt_file}: {e}")
                else:
                    print(f"No match found in 'subImages' for: {file}")



def rename_tif_by_adding_correct_gridNum(work_dir, reference_dir):
    regions_dirs = [dir for dir in  os.listdir(work_dir) if os.path.isdir(dir) ]
    for region_dir in regions_dirs:
        print(datetime.now(), f'working on {region_dir}')
        ref_dir = os.path.join(reference_dir, os.path.basename(region_dir))
        rename_tif_by_adding_correct_gridNum_one_region(region_dir, ref_dir)





def main():
    reference_dir = os.path.expanduser('~/Data/slump_demdiff_classify/clip_classify_extracted_traning_data/training_data')
    word_dir = os.path.expanduser('~/Data/slump_demdiff_classify/select_regions_Huangetal2023/training_data_manu_select')

    rename_tif_by_adding_correct_gridNum(word_dir, reference_dir)


if __name__ == '__main__':
    main()
