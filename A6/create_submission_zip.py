"""
create_submission_zip.py

Creates a zip version of the assignment directory to be submitted to Canvas.

Usage:
    1) Specify your SUNet ID in the variable provided below.
    2) Run this script from the root directory of the assignment to create a zip file
    named '<SUNet_ID>_submission.zip' in the assignment directory.

Run this script from the root directory of the assignment to create a zip file
named 'submission.zip' in the assignment directory. 

Example:
If you are in the root directory for assignment 3 (titled "A3"), run the following:
    python utils/zip/create_submission_zip.py
"""


SUNET_ID = "nallen2"

# =============================================================================
# You do not need to modify anything below this line.
# =============================================================================

import os
import zipfile

VALID_SOURCE_DIRECTORIES = ["A3", "A4", "A5", "A6"]


def file_filter(file: str):
    """
    Determines if a file should be included in the zip archive for submission.

    Parameters:
        file (str): The name of the file to check.

    Returns:
        bool: True if the file should be included, False otherwise.
    """

    # For python files
    if file.endswith(".py"):
        # Ensure that the file is not this script
        if file != "create_submission_zip.py":
            return True

    # For pngs
    if file.endswith(".png"):
        return True

    # For the ipynb
    if file.endswith(".ipynb"):
        return True

    # Otherwise
    return False


def create_submission_zip(source_directory: str, destination_zip: str):
    """
    Creates a zip version of the assignment directory to be submitted to GradeScope.

    Parameters:
        source_directory (str): The name of the directory to zip up.
        destination_zip (str): The name of the zip file to create.
    """
    with zipfile.ZipFile(destination_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_directory):
            for file in files:
                if file_filter(file):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(
                        file_path, source_directory
                    )  # Preserve organizational structure
                    zipf.write(file_path, arcname)


if __name__ == "__main__":
    # Use the current directory as the source directory
    source_directory = os.getcwd()

    # Determine if one of the VALID_SOURCE_DIRECTORIES is in the source directory
    if not any([source_directory.endswith(dir) for dir in VALID_SOURCE_DIRECTORIES]):
        raise ValueError(
            f"Please ensure you run this script from the root directory of the assignment. (e.g. 'A3', 'A4', 'A5', or 'A6')"
        )

    # Get the assignment name
    A_num = source_directory.split("/")[-1]

    if SUNET_ID == "":
        raise ValueError(
            "Please specify your SUNet ID in the variable provided in the script."
        )

    # Create the destination zip file in the source directory
    destination_zip = os.path.join(
        source_directory, f"{SUNET_ID}_submission_{A_num}.zip"
    )

    create_submission_zip(source_directory, destination_zip)
    print(
        f"Zip archive '{destination_zip}' created successfully in {source_directory}."
    )
