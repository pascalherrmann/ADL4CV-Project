import os
import config

def create_dir(path, exist_ok=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=exist_ok)
        print("Created Dir '{}'".format(path))

def write_to_file(content, path):
    file_object = open(path, 'a')
    file_object.write(content)
    file_object.close()

def convert_pickle_path_to_name(pickle_path):
    query = "run"
    s = pickle_path.find(query)
    sub_string = pickle_path[s:]
    replaced = sub_string.replace("/","_")
    return replaced




