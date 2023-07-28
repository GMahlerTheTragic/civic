import os
import sys


def get_project_root():
    # Get the absolute path of the main entry point (usually the script that is executed)
    main_script_path = os.path.abspath(sys.argv[0])

    # Go up the directory hierarchy until we find the file 'config.py' in the path
    current_path = main_script_path
    while not os.path.isfile(os.path.join(current_path, "config.py")):
        current_path = os.path.dirname(current_path)

    return current_path


# Call the function to get the project root and store it in a variable
PROJECT_ROOT = get_project_root()
