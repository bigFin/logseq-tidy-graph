# TODO    additional CLI commands (e.g., list_paths, extract_tags, estimate_cost as standalone commands)?
# TODO add error handling for invalid directories or user inputs?
import typer
from pathlib import Path
import questionary
from src.processor import process_graph, extract_tags, save_tags_to_file
from src.validator import validate_and_clean_paths, select_path
import tiktoken

# Define the Typer app
app = typer.Typer()

# Global default model
DEFAULT_MODEL = "gpt-4o-mini"

# Path to the paths file
PATHS_FILE = Path("./paths.txt")

# OpenAI model pricing
PRICING = {
    # Example pricing per 1K tokens
    "gpt-4o-mini": {"input": 0.015, "output": 0.025},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
}


def count_tokens(text: str, model: str) -> int:
    """
    Count the number of tokens in the text for a given model using tiktoken.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_cost(content_list: list, model: str = DEFAULT_MODEL, avg_output_tokens: int = 300) -> float:
    """
    Estimate the cost of processing a list of content chunks with OpenAI API.

    Args:
        content_list (list): List of text content (e.g., files or sections).
        model (str): Model name (e.g., "gpt-4o-mini", "gpt-3.5-turbo").
        avg_output_tokens (int): Estimated average output tokens per chunk.

    Returns:
        float: Total estimated cost in USD.
    """
    if model not in PRICING:
        raise ValueError(
            f"Pricing information for model {model} is not available.")

    input_token_cost = PRICING[model]["input"] / 1000  # Cost per token
    output_token_cost = PRICING[model]["output"] / 1000  # Cost per token

    total_input_tokens = sum(count_tokens(content, model)
                             for content in content_list)
    total_output_tokens = len(content_list) * avg_output_tokens

    total_cost = (total_input_tokens * input_token_cost) + \
        (total_output_tokens * output_token_cost)
    return total_cost


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
def tidy_graph(
    model: str = typer.Option(
        DEFAULT_MODEL, help="OpenAI model to use (e.g., gpt-4o-mini, gpt-3.5-turbo)")
):
    """
    CLI entry point to tidy a Logseq graph.
    """
    # Step 1: Validate paths in the file
    valid_paths = validate_and_clean_paths(PATHS_FILE)

    # Step 2: Let user choose a path or add a new one
    typer.echo("Select a Logseq graph:")
    if valid_paths:
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

    # Step 3: Extract tags and save them to a file
    typer.echo("Extracting unique #hashtags and [[backlinks]]...")
    tags = extract_tags(graph_path)
    tags_file = graph_path / "tags.txt"
    save_tags_to_file(tags, tags_file)
    typer.echo(f"Extracted tags have been saved to {tags_file}")

    # Step 4: Collect all content for cost estimation
    typer.echo("Collecting content for cost estimation...")
    journals_dir = graph_path / "journals"
    pages_dir = graph_path / "pages"
    content_list = []

    for file_path in journals_dir.glob("*.md"):
        content_list.append(file_path.read_text())
    for file_path in pages_dir.glob("*.md"):
        content_list.append(file_path.read_text())

    # Step 5: Estimate cost dynamically
    estimated_cost = estimate_cost(content_list, model=model)
    typer.echo(
        f"Estimated cost of processing this graph with {model}: ${estimated_cost:.2f}")

    confirm = typer.confirm("Do you want to proceed?")
    if not confirm:
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    # Step 6: Process the graph
    typer.echo("Processing the graph...")
    output_dir = typer.prompt(
        "Enter the output directory for the tidied graph")
    output_path = Path(output_dir).resolve()
    process_graph(graph_path, output_path, tags, model=model)
    typer.echo(f"Tidied graph has been saved to {output_path}")


if __name__ == "__main__":
    app()
