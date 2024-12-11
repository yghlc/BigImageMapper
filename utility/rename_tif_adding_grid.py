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
import rasterio

code_dir = os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS')
sys.path.insert(0, code_dir)
import basic_src.io_function as io_function
import basic_src.basic as basic

# def are_files_identical(file1, file2):
#     """
#     Check if two files are identical by comparing their sizes and contents.
#
#     Parameters:
#     file1 (str): Path to the first file.
#     file2 (str): Path to the second file.
#
#     Returns:
#     bool: True if the files are identical, False otherwise.
#     """
#     # First compare file sizes
#     if os.path.getsize(file1) != os.path.getsize(file2):
#         return False
#
#     # Compare file contents byte-by-byte
#     with open(file1, "rb") as f1, open(file2, "rb") as f2:
#         while True:
#             chunk1 = f1.read(8192)  # Read in chunks of 8 KB
#             chunk2 = f2.read(8192)
#             if chunk1 != chunk2:  # Compare the current chunks
#                 return False
#             if not chunk1:  # End of file reached
#                 break
#
#     return True


def are_files_identical(file1, file2):
    """
    Check if two .tif files are identical by comparing their width, height, band count, spatial coverage, and CRS.
    Outputs detailed information if the files differ.

    Parameters:
    file1 (str): Path to the first .tif file.
    file2 (str): Path to the second .tif file.

    Returns:
    bool: True if the files are identical, False otherwise.
    """
    differences = []
    try:
        with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
            # Compare width and height
            if src1.width != src2.width:
                differences.append(f"Width mismatch: {src1.width} (file1) != {src2.width} (file2)")
            if src1.height != src2.height:
                differences.append(f"Height mismatch: {src1.height} (file1) != {src2.height} (file2)")

            # Compare band count
            if src1.count != src2.count:
                differences.append(f"Band count mismatch: {src1.count} (file1) != {src2.count} (file2)")

            # Compare spatial coverage (bounds)
            if src1.bounds != src2.bounds:
                differences.append(f"Bounds mismatch:\n  file1: {src1.bounds}\n  file2: {src2.bounds}")

            # Compare coordinate reference systems (CRS)
            if src1.crs != src2.crs:
                differences.append(f"CRS mismatch:\n  file1: {src1.crs}\n  file2: {src2.crs}")

        if differences:
            print(f"Differences found between {file1} and {file2}:")
            for diff in differences:
                print(f"  - {diff}")
                basic.outputlogMessage(f"Differences found between {file1} and {file2}:")
                basic.outputlogMessage(f"  - {diff}")
            return False

        return True
    except Exception as e:
        print(f"Error comparing files {file1} and {file2}: {e}")
        # sys.exit(1)
        return False



def rename_tif_by_adding_correct_gridNum_one_region(region_dir, ref_dir):
    """
    Rename .tif files in subfolders of `region_dir` by matching their `_number.tif` suffix with files in `subImages` in `ref_dir`,
    only if the files are identical. Also, update the filenames in .txt files in subfolders of `region_dir`.

    Write by AI, modified by HLC

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

    # Open a log file to record mismatched files
    log_file_path = os.path.join(region_dir, "rename_mismatch.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("Mismatched Files Log:\n")

        # sub_dir_list = io_function.os_list_folder_dir(region_dir)
        sub_dir_list = [ os.path.join(region_dir,'subImages')]  # only work on subImages
        patch_list_txt = io_function.get_file_list_by_pattern(region_dir, '*patch_list.txt')
        for sub_dir in sub_dir_list:
            tif_list = io_function.get_file_list_by_pattern(sub_dir,'*.tif')
            for old_path in tif_list:
                file = os.path.basename(old_path)
                key = file.split("_")[-1]  # Extract `_number.tif`
                if key in sub_images_files.keys():
                    ref_file = sub_images_files[key]
                    ref_path = os.path.join(sub_images_dir, ref_file)

                    # Skip if the filenames are already the same
                    if file == ref_file:
                        # print(f"Skipping: {file} (already correctly named)")
                        continue

                    # Check if the files are identical
                    if not are_files_identical(old_path, ref_path):
                        # Log the mismatch
                        log_file.write(f"Mismatch: {file} and {ref_file}\n")
                        print(f"Mismatch logged: {file} and {ref_file}")

                    # Rename the .tif file
                    new_name = ref_file
                    new_path = os.path.join(sub_dir, new_name)
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {file} -> {new_name}")
                    except Exception as e:
                        print(f"Error renaming {file}: {e}")

                    # Update the filenames in .txt files in the same folder
                    for txt_path in patch_list_txt:
                            try:
                                with open(txt_path, "r") as f:
                                    content = f.read()

                                # Replace old filename with new filename
                                updated_content = content.replace(file, new_name)

                                with open(txt_path, "w") as f:
                                    f.write(updated_content)

                                print(f"Updated {txt_path}: {file} -> {new_name}")
                            except Exception as e:
                                print(f"Error updating {txt_path}: {e}")
                else:
                    print(f"No match found in 'subImages' for: {file}")

        print(f"Mismatch log saved to {log_file_path}")





def rename_tif_by_adding_correct_gridNum(work_dir, reference_dir):
    regions_dirs = [os.path.join(work_dir, dir) for dir in  os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, dir)) ]
    print('regions_dirs:')
    [print(item) for item in regions_dirs]
    for region_dir in regions_dirs:
        print(datetime.now(), f'working on {region_dir}')
        ref_folder_name = os.path.basename(region_dir).replace('_manu_Sel','')
        ref_dir = os.path.join(reference_dir, ref_folder_name)
        rename_tif_by_adding_correct_gridNum_one_region(region_dir, ref_dir)





def main():
    reference_dir = os.path.expanduser('~/Data/slump_demdiff_classify/clip_classify_extracted_traning_data/training_data')
    work_dir = os.path.expanduser('~/Data/slump_demdiff_classify/select_regions_Huangetal2023/training_data_manu_select')

    rename_tif_by_adding_correct_gridNum(work_dir, reference_dir)


if __name__ == '__main__':
    main()
