import typer
import asyncio
from pathlib import Path
import questionary
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.table import Table
from rich.console import Console
from src.processor import process_graph, extract_tags, save_tags_to_file
from src.validator import validate_and_clean_paths, select_path
import tiktoken
from typing import List, Tuple
import sys

app = typer.Typer()

DEFAULT_MODEL = "gpt-4o-mini"

PATHS_FILE = Path("./paths.txt")

PRICING = {
    "gpt-4o-mini": {"input": 0.015, "output": 0.025},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
}


def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_cost(content_list: list, model: str = DEFAULT_MODEL, avg_output_tokens: int = 300) -> float:
    if model not in PRICING:
        raise ValueError(
            "Pricing information for model {} is not available.".format(model))

    input_token_cost = PRICING[model]["input"] / 1000
    output_token_cost = PRICING[model]["output"] / 1000

    total_input_tokens = sum(count_tokens(content, model)
                             for content in content_list)
    total_output_tokens = len(content_list) * avg_output_tokens

    total_cost = (total_input_tokens * input_token_cost) + \
        (total_output_tokens * output_token_cost)
    return total_cost


def save_path_to_file(path: Path) -> None:
    path_str = str(path.resolve())
    try:
        existing_paths = set()
        if PATHS_FILE.exists():
            existing_paths = set(PATHS_FILE.read_text().splitlines())

        if path_str not in existing_paths:
            with PATHS_FILE.open('a') as f:
                f.write("{}\n".format(path_str))
    except Exception as e:
        typer.echo("Warning: Could not save path to file: {}".format(e), err=True)


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


async def process_files_with_progress(content_list: List[Tuple[str, Path]], output_path: Path, tags: set, model: str):
    console = Console()
    total_files = len(content_list)
    processed_files = 0
    failed_files = []

    progress_table = Table.grid()
    progress_table.add_row("[bold]Processing Logseq Graph Files")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "[cyan]Processing {} files...".format(total_files), total=total_files)

        batch_size = 20
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]

            try:
                batch_contents = [(content, tags) for content, _ in batch]
                async for result in tidy_content_batch_stream(batch_contents, model=model, batch_size=batch_size):
                    file_path = batch[processed_files][1]
                    output_file = output_path / \
                        file_path.relative_to(file_path.parent.parent)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    output_file.write_text(result)

                    processed_files += 1
                    progress.update(overall_task, advance=1)

            except Exception as e:
                failed_files.extend([f[1] for f in batch])
                console.print("[red]Error processing batch: {}".format(str(e)))
                processed_files += len(batch)
                progress.update(overall_task, advance=len(batch))

    console.print("\n[bold]Processing Complete!")
    console.print(
        "Successfully processed: {} files".format(processed_files - len(failed_files)))
    if failed_files:
        console.print(
            "[red]Failed to process {} files:".format(len(failed_files)))
        for file in failed_files:
            console.print("[red]  - {}".format(file))


@app.command("tidy-graph")
def tidy_graph(
    model: str = typer.Option(
        DEFAULT_MODEL, help="OpenAI model to use (e.g., gpt-4o-mini, gpt-3.5-turbo)")
):
    valid_paths = validate_and_clean_paths(PATHS_FILE)

    typer.echo("Select a Logseq graph:")
    if valid_paths:
        # Convert Path objects to strings and add the new path option
        choices = [str(path) for path in valid_paths] + ["[Enter a new path]"]
        selected = questionary.select(
            "Choose a graph or navigate to a new one:", choices=choices
        ).ask()
        if selected == "[Enter a new path]":
            graph_path = select_path_interactively()
            save_path_to_file(graph_path)
        else:
            graph_path = Path(selected)
    else:
        typer.echo("No valid paths found. Navigate to a Logseq graph.")
        graph_path = select_path_interactively()
        save_path_to_file(graph_path)

    typer.echo("Extracting unique #hashtags and [[backlinks]]...")
    tags = extract_tags(graph_path)
    tags_file = graph_path / "tags.txt"
    save_tags_to_file(tags, tags_file)
    typer.echo("Extracted tags have been saved to {}".format(tags_file))

    typer.echo("Collecting content for cost estimation...")
    journals_dir = graph_path / "journals"
    pages_dir = graph_path / "pages"
    content_list = []

    for file_path in journals_dir.glob("*.md"):
        content_list.append(file_path.read_text())
    for file_path in pages_dir.glob("*.md"):
        content_list.append(file_path.read_text())

    estimated_cost = estimate_cost(content_list, model=model)
    typer.echo(
        "Estimated cost of processing this graph with {}: ${:.2f}".format(model, estimated_cost))

    confirm = typer.confirm("Do you want to proceed?")
    if not confirm:
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    typer.echo("Processing the graph...")
    output_dir = typer.prompt(
        "Enter the output directory for the tidied graph")
    output_path = Path(output_dir).resolve()

    content_list = []
    for dir_name in ['journals', 'pages']:
        dir_path = graph_path / dir_name
        if dir_path.exists():
            for file_path in dir_path.glob("*.md"):
                content_list.append((file_path.read_text(), file_path))

    asyncio.run(process_files_with_progress(
        content_list, output_path, tags, model))
    typer.echo("Tidied graph has been saved to {}".format(output_path))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app()
