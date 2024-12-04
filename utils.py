import os
from pathlib import Path


def ensure_directory_exists(directory: Path):
    """
    Ensure a directory exists, creating it if necessary.
    """
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)


def read_file_content(file_path: Path) -> str:
    """
    Read and return the content of a file.
    """
    with file_path.open("r") as file:
        return file.read()


def write_file_content(file_path: Path, content: str):
    """
    Write content to a file.
    """
    with file_path.open("w") as file:
        file.write(content)


def count_files_in_directory(directory: Path, extension: str = "*.md") -> int:
    """
    Count the number of files with a given extension in a directory.
    """
    return len(list(directory.glob(extension)))


def extract_unique_items_from_list(items: list) -> list:
    """
    Return a sorted list of unique items.
    """
    return sorted(set(items))
