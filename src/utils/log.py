import os
from collections import defaultdict

def log_execution(base_dir, filename, log_txt, print_console=True):
    log_file_path = base_dir+'/'+filename

    try:
        os.makedirs(base_dir)
    except:
        pass

    with open(log_file_path, 'a') as file:
        file.write(log_txt)

    if(print_console):
        print(log_txt)
