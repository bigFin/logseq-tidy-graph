import typer
import asyncio
from pathlib import Path
import questionary
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.table import Table
from rich.console import Console
from src.processor import process_graph, extract_tags, save_tags_to_file
from src import tidy
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


def estimate_cost(content_list: List[Tuple[str, Path]], model: str = DEFAULT_MODEL, avg_output_tokens: int = 300) -> float:
    if model not in PRICING:
        raise ValueError(
            "Pricing information for model {} is not available.".format(model))

    input_token_cost = PRICING[model]["input"] / 1000
    output_token_cost = PRICING[model]["output"] / 1000

    total_input_tokens = sum(count_tokens(content, model)
                             for content, _ in content_list)
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


async def process_single_file(
    content: str,
    file_path: Path,
    output_path: Path,
    tags: set,
    model: str,
    progress,
    task_id,
    console: Console
) -> bool:
    """Process a single file and return True if successful."""
    try:
        # Update progress
        progress.update(
            task_id, description="Processing: {}".format(file_path.name))

        # Process the file
        async for result in tidy.tidy_content_batch_stream([(content, tags)], model=model, batch_size=1):
            if result:
                # Determine the output path maintaining directory structure
                if 'journals' in str(file_path):
                    rel_path = Path('journals') / file_path.name
                else:
                    rel_path = Path('pages') / file_path.name

                # Save the result
                output_file = output_path / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(result)

                console.print(
                    "[green]Successfully processed: {}".format(rel_path))
                progress.update(task_id, advance=1)
                return True

        raise RuntimeError("No result received for {}".format(file_path.name))

    except Exception as e:
        console.print("[red]Error processing {}: {}".format(file_path, str(e)))
        progress.update(task_id, advance=1)
        return False


async def process_files_with_progress(content_list: List[Tuple[str, Path]], output_path: Path, tags: set, model: str):
    """Process files with progress monitoring."""
    console = Console()
    total_files = len(content_list)
    successful_files = 0
    failed_files = []

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            "[cyan]Processing files...", total=total_files)

        batch_size = 20
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]

            # Process each file in the batch
            for content, file_path in batch:
                success = await process_single_file(
                    content=content,
                    file_path=file_path,
                    output_path=output_path,
                    tags=tags,
                    model=model,
                    progress=progress,
                    task_id=overall_task,
                    console=console
                )

                if success:
                    successful_files += 1
                else:
                    failed_files.append(file_path)

            # Check if user wants to continue after failures
            if len(failed_files) > 0 and (i + batch_size) < len(content_list):
                if not typer.confirm("\nContinue processing remaining files?", default=True):

                    raise typer.Abort("Processing cancelled by user")

        console.print("\n[bold]Processing Complete!")
        console.print(
            "Successfully processed: {} files".format(successful_files))

        if failed_files:
            console.print(
                "[red]Failed to process {} files:".format(len(failed_files)))
            for file in failed_files:
                console.print("[red]  - {}".format(file))

    if successful_files == 0:
        console.print("\n[red]Warning: No files were successfully processed!")


@app.command("tidy-graph")
def tidy_graph(
    model: str = typer.Option(
        DEFAULT_MODEL, help="OpenAI model to use (e.g., gpt-4o-mini, gpt-3.5-turbo)")
):
    valid_paths = validate_and_clean_paths(PATHS_FILE)

    typer.echo("Select a Logseq graph:")
    if valid_paths:
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

    for dir_path in [journals_dir, pages_dir]:
        if dir_path.exists():
            for file_path in dir_path.glob("*.md"):
                content_list.append((file_path.read_text(), file_path))

    total_files = len(content_list)
    sample_size = min(3, total_files)
    estimated_total_cost = estimate_cost(content_list, model=model)
    estimated_sample_cost = estimated_total_cost * (sample_size / total_files)

    typer.echo("\nEstimated costs:")
    typer.echo("Sample processing ({} files): ${:.3f}".format(
        sample_size, estimated_sample_cost))
    typer.echo("Full graph processing ({} files): ${:.2f}".format(
        total_files, estimated_total_cost))

    try_sample = typer.confirm(
        "\nWould you like to process a small sample first?")
    if try_sample:
        temp_output = Path("./sample_output")
        temp_output.mkdir(exist_ok=True)

        sample_content = []
        pages_sample = [c for c in content_list if 'pages' in str(
            c[1].parent)][:sample_size//2]
        journals_sample = [c for c in content_list if 'journals' in str(
            c[1].parent)][:sample_size//2]

        if not pages_sample and not journals_sample:
            sample_content = content_list[:sample_size]
        else:
            sample_content.extend(pages_sample)
            remaining_slots = sample_size - len(sample_content)
            if remaining_slots > 0:
                sample_content.extend(journals_sample[:remaining_slots])

        if not sample_content:
            sample_content = content_list[:sample_size]

        typer.echo("\nProcessing sample files...")
        asyncio.run(process_files_with_progress(
            sample_content, temp_output, tags, model))

        typer.echo("\nSample results have been saved to ./sample_output")
        typer.echo(
            "Please review the processed files and decide if you want to continue.")

        if typer.confirm("Would you like to view the processed sample files?"):
            successful_files = []
            for _, file_path in sample_content:
                sample_file = temp_output / \
                    file_path.relative_to(file_path.parent.parent)
                if sample_file.exists():
                    successful_files.append(sample_file)
                    typer.launch(str(sample_file))

            if not successful_files:
                typer.echo(
                    "[red]No successfully processed files found to view.")
            else:
                typer.echo("Opened {} processed files for review.".format(
                    len(successful_files)))

        proceed = typer.confirm(
            "\nWould you like to proceed with processing the entire graph?")
        if not proceed:
            typer.echo("Operation cancelled.")
            raise typer.Exit()

    else:
        proceed = typer.confirm(
            "\nWould you like to proceed with processing the entire graph?")
        if not proceed:
            typer.echo("Operation cancelled.")
            raise typer.Exit()

    typer.echo("\nProcessing the graph...")
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
