import os


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist - creating it")
        try:
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"An error occurred while creating the folder: {e}")


def check_file_exists(path_to_file):
    return os.path.exists(path_to_file)
