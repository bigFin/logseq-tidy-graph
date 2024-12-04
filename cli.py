# TODO    additional CLI commands (e.g., list_paths, extract_tags, estimate_cost as standalone commands)?
# TODO add error handling for invalid directories or user inputs?


import os
from pathlib import Path
import typer
from src.validator import validate_and_clean_paths, add_new_path, select_path
from src.processor import process_graph, extract_tags, save_tags_to_file, estimate_cost

app = typer.Typer()

# Path to the paths file
PATHS_FILE = Path("./paths.txt")


@app.command()
def tidy_graph():
    """
    CLI entry point to tidy a Logseq graph.
    """
    # Step 1: Validate paths in the file
    valid_paths = validate_and_clean_paths(PATHS_FILE)

    # Step 2: Let user choose a path or add a new one
    graph_path = select_path(valid_paths, PATHS_FILE)

    # Step 3: Extract tags and save them to a file
    typer.echo("Extracting unique #hashtags and [[backlinks]]...")
    tags = extract_tags(graph_path)
    tags_file = graph_path / "tags.txt"
    save_tags_to_file(tags, tags_file)
    typer.echo(f"Extracted tags have been saved to {tags_file}")

    # Step 4: Estimate cost
    estimated_cost = estimate_cost(graph_path)
    typer.echo(
        f"Estimated cost of processing this graph: ${estimated_cost:.2f}")
    confirm = typer.confirm("Do you want to proceed?")
    if not confirm:
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    # Step 5: Process the graph
    output_dir = typer.prompt(
        "Enter the output directory for the tidied graph")
    output_path = Path(output_dir).resolve()
    process_graph(graph_path, output_path, tags)
    typer.echo(f"Tidied graph has been saved to {output_path}")


if __name__ == "__main__":
    app()
