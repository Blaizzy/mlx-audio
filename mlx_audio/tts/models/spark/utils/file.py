# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Description:
    This script contains a collection of functions designed to handle various
    file reading and writing operations. It provides utilities to read from files,
    write data to files, and perform file manipulation tasks.
"""


import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def resolve_symbolic_link(symbolic_link_path: Path) -> Path:
    """
    Resolves the absolute path of a symbolic link.

    Args:
        symbolic_link_path (Path): The path to the symbolic link.

    Returns:
        Path: The absolute path that the symbolic link points to.
    """

    link_directory = os.path.dirname(symbolic_link_path)
    target_path_relative = os.readlink(symbolic_link_path)
    return os.path.join(link_directory, target_path_relative)


def write_jsonl(metadata: List[dict], file_path: Path) -> None:
    """Writes a list of dictionaries to a JSONL file.

    Args:
    metadata : List[dict]
        A list of dictionaries, each representing a piece of meta.
    file_path : Path
        The file path to save the JSONL file

    This function writes each dictionary in the list to a new line in the specified file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for meta in tqdm(metadata, desc="writing jsonl"):
            # Convert dictionary to JSON string and write it to the file with a newline
            json_str = json.dumps(meta, ensure_ascii=False) + "\n"
            f.write(json_str)
    print(f"jsonl saved to {file_path}")


def read_jsonl(file_path: Path) -> List[dict]:
    """
    Reads a JSONL file and returns a list of dictionaries.

    Args:
    file_path : Path
        The path to the JSONL file to be read.

    Returns:
    List[dict]
        A list of dictionaries parsed from each line of the JSONL file.
    """
    metadata = []
    # Open the file for reading
    with open(file_path, "r", encoding="utf-8") as f:
        # Split the file into lines
        lines = f.read().splitlines()
    # Process each line
    for line in lines:
        # Convert JSON string back to dictionary and append to list
        meta = json.loads(line)
        metadata.append(meta)
    # Return the list of metadata
    return metadata


def read_json_as_jsonl(file_path: Path) -> List[dict]:
    metadata = []
    with open(file_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    for k in sorted(data.keys()):
        meta = {"index": k}
        meta.update(data[k])
        metadata.append(meta)
    return metadata


def decode_unicode_strings(meta: Dict[str, Any]) -> Dict[str, Any]:
    processed_meta = {}
    for k, v in meta.items():
        if isinstance(v, str):
            processed_meta[k] = v.encode("utf-8").decode("unicode_escape")
        else:
            processed_meta[k] = v
    return processed_meta


def load_config(config_path: Path) -> DictConfig:
    """Loads a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config


def jsonl_to_csv(jsonl_file_path: str, csv_file_path: str) -> None:
    """
    Converts a JSONL file to a CSV file.

    This function reads a JSONL file, determines all unique keys present in the file,
    and writes the data to a CSV file with columns for all these keys.
    """

    all_keys = set()
    data_rows = []

    # Read the JSONL file once to extract keys and collect data
    with open(jsonl_file_path, "r") as file:
        for line in file:
            data = json.loads(line.strip())
            data_rows.append(data)
            all_keys.update(data.keys())

    # Convert the set of keys to a sorted list for consistent column order
    sorted_keys = sorted(all_keys)

    # Write the data to a CSV file
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted_keys)

        # Write the header row
        writer.writeheader()

        # Write each row of data
        for data in data_rows:
            writer.writerow(data)

    print(f"CSV file has been created at {csv_file_path}")


def save_metadata(data, filename, headers=None):
    """
    Save metadata to a file.

    Args:
        data (list of dict): Metadata to be saved.
        filename (str): Name of the file to save the metadata.
        headers (list of str): The order of column names to be saved; defaults to the keys from the first dictionary in data if not provided.
    """
    # Set headers to keys from the first dictionary in data if not explicitly provided
    if headers is None:
        headers = list(data[0].keys())

    with open(filename, "w", encoding="utf-8") as file:
        # Write the headers to the file
        file.write("|".join(headers) + "\n")
        for entry in data:
            # Retrieve values in the order of headers, replacing any '|' characters with a space to prevent formatting errors
            formatted_values = [
                str(entry.get(key, "")).replace("|", " ") for key in headers
            ]
            # Write the formatted values to the file
            file.write("|".join(formatted_values) + "\n")


def read_metadata(filename, headers=None):
    """
    Read metadata from a file.

    Args:
        filename (str): The file from which to read the metadata.

    Returns:
        list of dict: The metadata read from the file.
        list of str: The headers used in the file.
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    data = []
    # Set headers from the first line of the file if not provided
    if headers is None:
        headers = lines[0].strip().split("|")
        lines = lines[1:]

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Split the line by '|' and pair with headers to form a dictionary
        entry_data = dict(zip(headers, line.split("|")))
        data.append(entry_data)

    return data, headers
