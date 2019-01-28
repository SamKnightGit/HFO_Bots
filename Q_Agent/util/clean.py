import subprocess
import os


def clean_keep_lines(logging_directory, logs, num_lines):

    temppath = logging_directory + '/tempfile.txt'

    for log_path in logs:
        with open(temppath, 'w+') as tempfile:
            subprocess.run(['tail', '-n', str(num_lines), log_path], stdout=tempfile)
        subprocess.run(['mv', temppath, log_path])


def clean_dir_keep_lines(logging_directory, num_lines):

    files_in_dir = []

    for file in os.listdir(logging_directory):
        files_in_dir.append(os.path.abspath(file))

    clean_keep_lines(logging_directory, files_in_dir, num_lines)