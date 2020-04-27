import os
import pickle
import pandas as pd
from  musket2.context import  get_current_project_data_path
import  imageio
import yaml
from threading import Lock

def load(path):
    with open(path, "rb") as f:
        return pickle.load(f);


def save(path,data):
    with open(path, "wb") as f:
        pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)


def delete_file(path:str, recursively = True):
    if os.path.isdir(path):
        if recursively:
            for chPath in (os.path.join(path, f) for f in os.listdir(path)):
                delete_file(chPath, True)
        os.rmdir(path)
    else:
        os.unlink(path)

def ensure(directory):
    try:
        os.makedirs(directory)
    except:
        pass


def csv_from_data(relative_path:str):
    if isinstance(relative_path, pd.DataFrame):
        return relative_path
    try:
        return pd.read_csv(os.path.join(get_current_project_data_path(),relative_path))
    except:
        return pd.read_csv(relative_path)


def either(opt0,opt1):
    if opt0 is not None:
        return opt0
    return  opt1

def key_or_default(key:str,d:dict,opt1):
    if key in d:
        return d[key]
    return opt1



def image_from_data(relative_path:str):
    return imageio.imread(os.path.join(get_current_project_data_path(),relative_path))

_l=Lock()


def load_yaml(path):
    _l.acquire()
    try:
        yaml_load = lambda x: yaml.load(x, Loader=yaml.Loader)

        with open(path, "r") as f:
            return yaml_load(f);
    finally:
        _l.release()


def load_string(path):
    with open(path, 'r') as myfile:
        data = myfile.read()
        return data


def save_string(path,data):
    with open(path, 'w') as myfile:
        myfile.write(data)


def save_yaml(path, data, header=None):
    _l.acquire()
    try:
        with open(path, "w") as f:
            if header:
                text = yaml.dump(data)

                text = header + "\n" + text;

                f.write(text)

                return None
            return yaml.dump(data, f)
    finally:
        _l.release()


def normalize_path(pth:str)->str:
    project_local=os.path.join(get_current_project_data_path(), pth)
    if os.path.exists(project_local):
        return project_local
    return pth