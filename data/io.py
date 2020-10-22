"""
Optimised methods for large scale data handling.
"""
import json
import pathlib
import pickle

import h5py


def save_array_to_hdf5(dirpath, filename, array):
    """
    Save a Numpy Array to HDF5 format.
    Parameters:
    -----------
    dirname : str
    filename : str
    array : np.ndarray
    Returns:
    --------
    None
    """
    dirpath = pathlib.Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    filepath = dirpath / f'{filename}.hdf5'

    with h5py.File(filepath, 'a') as f:
        f.create_dataset(f"{filename}", data=array)


def load_hdf5(filename, dirpath):
    """
    Load HDF5 file from disk.
    Parameters:
    -----------
    filename : str
    dirpath : str
    Returns:
    --------
    h5py.File
    """
    dirpath = pathlib.Path(dirpath)
    filepath = dirpath / f'{filename}.hdf5'

    return h5py.File(filepath, 'r')


def load_hdf5_to_array(dataname, filename, dirpath):
    """
    Load HDF5 file from disk into an Numpy array object.
    Parameters:
    ----------
    dataname : str
        HDF5 object data name
    filename : str
    dirpath : str
    Returns:
    --------
    np.ndarray
    """

    hdf5_file = load_hdf5(filename, dirpath)

    return hdf5_file[dataname][:]


def load_json(filepath):
    """
    Load json into dictionary.
    Parameters:
    -----------
    filename : str
    directory : str
    Returns:
    --------
    dict
    """

    with open(filepath, 'r') as f:
        obj = json.load(f)
    return obj


def file_in_directory(filename, dirpath, ext='hdf5'):
    """
    Check if a file with a given name already exists in a given directory.
    Parameters:
    -----------
    filename : str
    dirpath: str
    Returns:
    --------
    bool
    """
    dirpath = pathlib.Path(dirpath).glob(f'*.{ext}')

    files = [f for f in dirpath if f.is_file()]

    for file_ in files:
        if filename in file_.name:
            return True
    return False


def directory_exists(dirpath):
    """
    Check if directory at dirpath exists.
    Parameters:
    -----------
    dirpath : str
    Returns:
    --------
    bool
    """
    dirpath = pathlib.Path(dirpath)

    if dirpath.is_dir():
        return True
    return False
