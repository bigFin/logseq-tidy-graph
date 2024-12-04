import os
from pathlib import Path
import typer


def validate_and_clean_paths(paths_file: Path) -> list:
    """
    Validate paths from the paths file and remove invalid ones.
    """
    valid_paths = []
    if paths_file.exists():
        with paths_file.open("r") as file:
            for line in file:
                path = Path(line.strip())
                if path.exists() and (path / "journals").is_dir() and (path / "pages").is_dir():
                    valid_paths.append(path)
                else:
                    typer.echo(f"Invalid path removed: {path}")
    with paths_file.open("w") as file:
        file.writelines(f"{p}\n" for p in valid_paths)
    return valid_paths


def add_new_path(new_path: str, paths_file: Path) -> Path:
    """
    Add a new path to the paths file if valid.
    """
    path = Path(new_path).resolve()
    if not path.exists() or not (path / "journals").is_dir() or not (path / "pages").is_dir():
        raise ValueError(f"Path {path} is not a valid Logseq graph.")
    with paths_file.open("a") as file:
        file.write(f"{path}\n")
    return path


def select_path(valid_paths: list, paths_file: Path) -> Path:
    """
    Let the user select a path or input a new one.
    """
    typer.echo("Select a Logseq graph:")
    for i, path in enumerate(valid_paths, start=1):
        typer.echo(f"{i}: {path}")
    typer.echo(f"{len(valid_paths) + 1}: Enter a new path")
    choice = typer.prompt("Enter your choice", type=int)

    if choice == len(valid_paths) + 1:
        new_path = typer.prompt("Enter the new graph path")
        return add_new_path(new_path, paths_file)
    return valid_paths[choice - 1]
