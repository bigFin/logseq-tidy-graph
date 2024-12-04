import aiohttp
import json
from typing import Dict, Optional
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

CONFIG_DIR = Path("config")
PRICING_FILE = CONFIG_DIR / "model_pricing.json"


async def fetch_model_pricing() -> Dict:
    """Fetch current model pricing from OpenAI API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.openai.com/v1/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    return {}
    except Exception as e:
        console = Console()
        console.print(
            "[yellow]Warning: Could not fetch model pricing: {}".format(e))
    return {}


def load_pricing() -> Dict:
    """Load pricing from local config file, creating default if needed."""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True)

    if not PRICING_FILE.exists():
        default_pricing = {
            "gpt-4o": {
                "input": 2.50,
                "input_batch": 1.25,
                "input_cached": 1.25,
                "output": 10.00,
                "output_batch": 5.00
            },
            "gpt-4o-2024-11-20": {
                "input": 2.50,
                "input_batch": 1.25,
                "input_cached": 1.25,
                "output": 10.00,
                "output_batch": 5.00
            },
            "gpt-4o-2024-08-06": {
                "input": 2.50,
                "input_batch": 1.25,
                "input_cached": 1.25,
                "output": 10.00,
                "output_batch": 5.00
            },
            "gpt-4o-2024-05-13": {
                "input": 5.00,
                "input_batch": 2.50,
                "output": 15.00,
                "output_batch": 7.50
            },
            "gpt-4o-mini": {
                "input": 0.150,
                "input_batch": 0.075,
                "input_cached": 0.075,
                "output": 0.600,
                "output_batch": 0.300
            },
            "gpt-4o-mini-2024-07-18": {
                "input": 0.150,
                "input_batch": 0.075,
                "input_cached": 0.075,
                "output": 0.600,
                "output_batch": 0.300
            },
            "o1-mini": {
                "input": 3.00,
                "input_cached": 1.50,
                "output": 12.00
            },
            "o1-mini-2024-09-12": {
                "input": 3.00,
                "input_cached": 1.50,
                "output": 12.00
            }
        }
        PRICING_FILE.write_text(json.dumps(default_pricing, indent=2))
        return default_pricing

    try:
        return json.loads(PRICING_FILE.read_text())
    except Exception as e:
        console = Console()
        console.print(
            "[yellow]Warning: Error loading pricing file: {}".format(e))
        return {}


def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_cost(
    content_list: List[Tuple[str, Path]],
    model: str = DEFAULT_MODEL,
    avg_output_tokens: int = 300,
    use_batch: bool = True,
    assume_cached: bool = False
) -> dict:
    """
    Estimate processing costs with detailed breakdown.
    Returns a dictionary with cost details.
    """
    pricing = load_pricing()

    if model not in pricing:
        raise ValueError(
            "Pricing information for model {} is not available. Please update {} with current pricing.".format(
                model, PRICING_FILE)
        )

    model_pricing = pricing[model]

    if assume_cached and "input_cached" in model_pricing:
        input_token_cost = model_pricing["input_cached"] / 1_000_000
    elif use_batch and "input_batch" in model_pricing:
        input_token_cost = model_pricing["input_batch"] / 1_000_000
    else:
        input_token_cost = model_pricing["input"] / 1_000_000

    output_token_cost = (
        model_pricing["output_batch"] / 1_000_000 if use_batch and "output_batch" in model_pricing
        else model_pricing["output"] / 1_000_000
    )

    total_input_tokens = sum(count_tokens(content, model)
                             for content, _ in content_list)
    total_output_tokens = len(content_list) * avg_output_tokens

    input_cost = total_input_tokens * input_token_cost
    output_cost = total_output_tokens * output_token_cost
    total_cost = input_cost + output_cost

    return {
        "total_cost": total_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "pricing_type": (
            "cached" if assume_cached else
            "batch" if use_batch else
            "standard"
        )
    }


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
        progress.update(
            task_id, description="Processing: {}".format(file_path.name))

        async for result in tidy.tidy_content_batch_stream([(content, tags)], model=model, batch_size=1):
            if result:
                if 'journals' in str(file_path):
                    rel_path = Path('journals') / file_path.name
                else:
                    rel_path = Path('pages') / file_path.name

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

    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        overall_progress = progress.add_task(
            "[bold blue]Overall Progress", total=total_files)
        file_progress = progress.add_task(
            "[cyan]Current File", total=total_files, visible=True)

        batch_size = 20
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]

            tasks = []
            for content, file_path in batch:
                task = process_single_file(
                    content=content,
                    file_path=file_path,
                    output_path=output_path,
                    tags=tags,
                    model=model,
                    progress=progress,
                    task_id=file_progress,
                    console=console
                )
                tasks.append((task, file_path))

            results = await asyncio.gather(*(task for task, _ in tasks), return_exceptions=True)

            for (_, file_path), result in zip(tasks, results):
                if isinstance(result, Exception):
                    console.print(
                        f"[red]Error processing {file_path}: {str(result)}")
                    failed_files.append(file_path)
                elif result:
                    successful_files += 1
                else:
                    failed_files.append(file_path)

                progress.update(overall_progress, advance=1)

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
        DEFAULT_MODEL, help="OpenAI model to use (e.g., gpt-4o-mini, gpt-3.5-turbo)"),
    update_pricing: bool = typer.Option(
        False, "--update-pricing", help="Update model pricing information")
):
    """Process and tidy a Logseq graph."""
    if update_pricing:
        asyncio.run(fetch_model_pricing())
        typer.echo("Model pricing information updated.")

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

    standard_cost = estimate_cost(
        content_list, model=model, use_batch=False, assume_cached=False)
    batch_cost = estimate_cost(
        content_list, model=model, use_batch=True, assume_cached=False)
    cached_cost = estimate_cost(
        content_list, model=model, use_batch=True, assume_cached=True)

    sample_ratio = sample_size / total_files
    sample_standard = {k: v * sample_ratio if isinstance(v, (int, float)) else v
                       for k, v in standard_cost.items()}
    sample_batch = {k: v * sample_ratio if isinstance(v, (int, float)) else v
                    for k, v in batch_cost.items()}

    pricing = load_pricing()

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
