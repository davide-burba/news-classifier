from datetime import datetime as dt
import os


def get_time():
    """
    Return current UTC time as string in format `%Y-%m-%d_%Hh%Mm%Ss`
    """
    return dt.utcnow().strftime("%Y-%m-%d_%Hh%Mm%Ss")


def build_output_dir(path_dir):
    """
    Build UTC timestamped subdir in path_dir, return path (str).
    """
    path_dir = f"{path_dir}/run_{get_time()}"
    os.makedirs(path_dir)
    return path_dir
