# TODO    additional CLI commands (e.g., list_paths, extract_tags, estimate_cost as standalone commands)?
# TODO add error handling for invalid directories or user inputs?


import questionary
from pathlib import Path
import typer
from src.processor import process_graph, extract_tags, save_tags_to_file, estimate_cost
from src.validator import validate_and_clean_paths

app = typer.Typer()

# Path to the paths file
PATHS_FILE = Path("./paths.txt")


def select_path_interactively(start_dir: Path = Path(".")) -> Path:
    """
    Use `questionary` to interactively traverse the filesystem and select a path.
    """
    current_dir = start_dir.resolve()

    while True:
        # List directories and files
        choices = [".. (Go up one level)"] + [
            f"{item.name}/" if item.is_dir() else item.name
            for item in sorted(current_dir.iterdir())
        ]

        # Prompt the user for selection
        choice = questionary.select(
            f"Current directory: {current_dir}",
            choices=choices + ["[Select this directory]"],
        ).ask()

        if choice == ".. (Go up one level)":
            # Move up one directory
            current_dir = current_dir.parent
        elif choice == "[Select this directory]":
            # Return the selected directory
            return current_dir
        else:
            # Move into the selected subdirectory or select a file
            selected = current_dir / choice.rstrip("/")
            if selected.is_dir():
                current_dir = selected
            else:
                typer.echo("Please select a directory, not a file.")


@app.command("tidy-graph")
def tidy_graph():
    """
    CLI entry point to tidy a Logseq graph.
    """
    typer.echo("Validating paths...")
    valid_paths = validate_and_clean_paths(PATHS_FILE)

    # Let user select or navigate to a new path
    if valid_paths:
        typer.echo("Select a Logseq graph:")
        valid_paths.append("[Enter a new path]")
        selected = questionary.select(
            "Choose a graph or navigate to a new one:", choices=valid_paths
        ).ask()
        if selected == "[Enter a new path]":
            graph_path = select_path_interactively()
        else:
            graph_path = Path(selected)
    else:
        typer.echo("No valid paths found. Navigate to a Logseq graph.")
        graph_path = select_path_interactively()

    # Extract tags and save them
    typer.echo("Extracting unique #hashtags and [[backlinks]]...")
    tags = extract_tags(graph_path)
    tags_file = graph_path / "tags.txt"
    save_tags_to_file(tags, tags_file)
    typer.echo(f"Extracted tags have been saved to {tags_file}")

    # Estimate cost
    estimated_cost = estimate_cost(graph_path)
    typer.echo(
        f"Estimated cost of processing this graph: ${estimated_cost:.2f}")
    confirm = typer.confirm("Do you want to proceed?")
    if not confirm:
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    # Process the graph
    output_dir = typer.prompt(
        "Enter the output directory for the tidied graph")
    output_path = Path(output_dir).resolve()
    process_graph(graph_path, output_path, tags)
    typer.echo(f"Tidied graph has been saved to {output_path}")


if __name__ == "__main__":
    app()
