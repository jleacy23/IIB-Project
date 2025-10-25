import numpy as np
import h5py
from typing import Tuple
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"/ "temp"

def write_input_data_hdf5(file_name: str, data: Tuple[np.ndarray, np.ndarray], dataset_name: str = "input_data") -> None:
    """ Writes fixed-point real and imaginary data to an HDF5 file for processing in C++"""
    file_path = DATA_DIR / file_name

    with h5py.File(file_path, 'w') as h5f:
        real_data, imag_data = data
        h5f.create_dataset(f"{dataset_name}/real", data=real_data)
        h5f.create_dataset(f"{dataset_name}/imag", data=imag_data)

def read_output_data_hdf5(file_name: str, dataset_name: str = "output_data") -> Tuple[np.ndarray, np.ndarray]:
    """ Reads fixed-point real and imaginary data from an HDF5 file produced by C++"""
    file_path = DATA_DIR / file_name
    with h5py.File(file_path, 'r') as h5f:
        real_data = h5f[f"{dataset_name}/real"][:]
        imag_data = h5f[f"{dataset_name}/imag"][:]
    return real_data, imag_data

def float_to_fixed_point(value: float, word_length: int, frac: int) -> int:
    """ Converts a float to its fixed-point representation"""
    scale = 1 << frac
    max_val = (1 << (word_length - 1)) - 1
    min_val = -(1 << (word_length - 1))
    fixed_point_value = int(round(value * scale))
    if fixed_point_value > max_val or fixed_point_value < min_val:
        raise ValueError("Value out of range for the specified fixed-point format")
    return fixed_point_value & ((1 << word_length) - 1)

def fixed_point_to_float(value: int, word_length: int, frac: int) -> float:
    """ Converts a fixed-point representation back to float"""
    scale = 1 << frac
    if value & (1 << (word_length - 1)):
        value -= (1 << word_length)
    return float(value) / scale

def complex_symbols_to_fixed_point(symbols: np.ndarray, word_length: int, frac: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Converts complex symbols to fixed-point representation"""
    real_parts, imag_parts = symbols.real, symbols.imag 
    real_fixed = np.array([float_to_fixed_point(r, word_length, frac) for r in real_parts])
    imag_fixed = np.array([float_to_fixed_point(i, word_length, frac) for i in imag_parts])
    return real_fixed, imag_fixed

def fixed_point_to_complex_symbols(real_fixed: np.ndarray, imag_fixed: np.ndarray, word_length: int, frac: int) -> np.ndarray:
    """ Converts fixed-point representation back to complex symbols"""
    real_parts = np.array([fixed_point_to_float(r, word_length, frac) for r in real_fixed])
    imag_parts = np.array([fixed_point_to_float(i, word_length, frac) for i in imag_fixed])
    return real_parts + 1j * imag_parts


