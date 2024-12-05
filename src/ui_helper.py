from pathlib import Path
import questionary
import typer
from datetime import datetime


def save_path_to_file(path: Path, paths_file: Path) -> None:
    path_str = str(path.resolve())
    try:
        existing_paths = set()
        if paths_file.exists():
            existing_paths = set(paths_file.read_text().splitlines())

        if path_str not in existing_paths:
            with paths_file.open('a') as f:
                f.write("{}\n".format(path_str))
    except Exception as e:
        typer.echo("Warning: Could not save path to file: {}".format(e), err=True)


def generate_output_name(input_path: Path) -> str:
    """Generate output graph name based on input path and current datetime."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    input_name = input_path.name
    return f"{input_name}_tidy_{timestamp}"


def select_path_interactively(start_dir: Path = Path(".")) -> Path:
    current_dir = start_dir.resolve()

    while True:
        choices = [".. (Go up one level)"] + [
            "{}/".format(item.name) if item.is_dir() else item.name
            for item in sorted(current_dir.iterdir())
        ]

        choice = questionary.select(
            "Current directory: {}".format(current_dir),
            choices=choices + ["[Select this directory]"],
        ).ask()

        if choice == ".. (Go up one level)":
            current_dir = current_dir.parent
        elif choice == "[Select this directory]":
            return current_dir
        else:
            selected = current_dir / choice.rstrip("/")
            if selected.is_dir():
                current_dir = selected
            else:
                typer.echo("Please select a directory, not a file.")


def display_cost_estimates(standard_cost: dict, batch_cost: dict, cached_cost: dict,
                           total_files: int, sample_size: int, model: str, pricing: dict):
    """Display cost estimates in a formatted way."""
    sample_ratio = sample_size / total_files
    sample_standard = {k: v * sample_ratio if isinstance(v, (int, float)) else v
                       for k, v in standard_cost.items()}
    sample_batch = {k: v * sample_ratio if isinstance(v, (int, float)) else v
                    for k, v in batch_cost.items()}

    typer.echo("\nEstimated costs:")
    typer.echo("\nSample processing ({} files):".format(sample_size))
    typer.echo("  Standard: ${:.3f}".format(sample_standard["total_cost"]))
    if model in pricing and "input_batch" in pricing[model]:
        typer.echo("  With Batch API: ${:.3f}".format(
            sample_batch["total_cost"]))

    typer.echo("\nFull graph processing ({} files):".format(total_files))
    typer.echo("  Standard: ${:.2f}".format(standard_cost["total_cost"]))
    if model in pricing and "input_batch" in pricing[model]:
        typer.echo("  With Batch API: ${:.2f}".format(
            batch_cost["total_cost"]))
        if "input_cached" in pricing[model]:
            typer.echo(
                "  With Batch API + Cache: ${:.2f}".format(cached_cost["total_cost"]))

    typer.echo("\nToken Usage:")
    typer.echo("  Input tokens: {:,}".format(
        standard_cost["total_input_tokens"]))
    typer.echo("  Estimated output tokens: {:,}".format(
        standard_cost["total_output_tokens"]))


def save_path_to_file(path: Path, paths_file: Path) -> None:
    path_str = str(path.resolve())
    try:
        existing_paths = set()
        if paths_file.exists():
            existing_paths = set(paths_file.read_text().splitlines())

        if path_str not in existing_paths:
            with paths_file.open('a') as f:
                f.write("{}\n".format(path_str))
    except Exception as e:
        typer.echo("Warning: Could not save path to file: {}".format(e), err=True)


def generate_output_name(input_path: Path) -> str:
    """Generate output graph name based on input path and current datetime."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    input_name = input_path.name
    return f"{input_name}_tidy_{timestamp}"


def select_path_interactively(start_dir: Path = Path(".")) -> Path:
    current_dir = start_dir.resolve()

    while True:
        choices = [".. (Go up one level)"] + [
            "{}/".format(item.name) if item.is_dir() else item.name
            for item in sorted(current_dir.iterdir())
        ]

        choice = questionary.select(
            "Current directory: {}".format(current_dir),
            choices=choices + ["[Select this directory]"],
        ).ask()

        if choice == ".. (Go up one level)":
            current_dir = current_dir.parent
        elif choice == "[Select this directory]":
            return current_dir
        else:
            selected = current_dir / choice.rstrip("/")
            if selected.is_dir():
                current_dir = selected
            else:
                typer.echo("Please select a directory, not a file.")


def display_cost_estimates(standard_cost: dict, batch_cost: dict, cached_cost: dict,
                           total_files: int, sample_size: int, model: str, pricing: dict):
    """Display cost estimates in a formatted way."""
    sample_ratio = sample_size / total_files
    sample_standard = {k: v * sample_ratio if isinstance(v, (int, float)) else v
                       for k, v in standard_cost.items()}
    sample_batch = {k: v * sample_ratio if isinstance(v, (int, float)) else v
                    for k, v in batch_cost.items()}

    typer.echo("\nEstimated costs:")
    typer.echo("\nSample processing ({} files):".format(sample_size))
    typer.echo("  Standard: ${:.3f}".format(sample_standard["total_cost"]))
    if model in pricing and "input_batch" in pricing[model]:
        typer.echo("  With Batch API: ${:.3f}".format(
            sample_batch["total_cost"]))

    typer.echo("\nFull graph processing ({} files):".format(total_files))
    typer.echo("  Standard: ${:.2f}".format(standard_cost["total_cost"]))
    if model in pricing and "input_batch" in pricing[model]:
        typer.echo("  With Batch API: ${:.2f}".format(
            batch_cost["total_cost"]))
        if "input_cached" in pricing[model]:
            typer.echo(
                "  With Batch API + Cache: ${:.2f}".format(cached_cost["total_cost"]))

    typer.echo("\nToken Usage:")
    typer.echo("  Input tokens: {:,}".format(
        standard_cost["total_input_tokens"]))
    typer.echo("  Estimated output tokens: {:,}".format(
        standard_cost["total_output_tokens"]))
