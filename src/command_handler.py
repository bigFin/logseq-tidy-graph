import typer
import asyncio
from pathlib import Path
from typing import List, Tuple
import questionary
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from .ui_helper import (
    save_path_to_file, generate_output_name,
    select_path_interactively, display_cost_estimates
)
from .file_processor import (
    process_files_with_progress, collect_content_list,
    copy_assets_folder
)
from .cost_manager import (
    RateLimiter, fetch_model_pricing, load_pricing,
    estimate_cost
)
from .processor import process_graph, extract_tags, save_tags_to_file

from .validator import validate_and_clean_paths

DEFAULT_MODEL = "gpt-4o-mini"


async def process_sample(
    content_list: List[Tuple[str, Path]],
    sample_size: int,
    tags: set,
    model: str
) -> bool:
    """Process a sample of files and return whether to continue."""
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
    rate_limiter = RateLimiter(500, 150000)  # RPM and TPM limits
    await process_files_with_progress(
        content_list=sample_content,
        output_path=temp_output,
        tags=tags,
        model=model,
        rate_limiter=rate_limiter
    )

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
            typer.echo("[red]No successfully processed files found to view.")
        else:
            typer.echo("Opened {} processed files for review.".format(
                len(successful_files)))

    return typer.confirm("\nWould you like to proceed with processing the entire graph?")


def handle_tidy_graph_command(
    model: str = DEFAULT_MODEL,
    update_pricing: bool = False,
    input_paths_file: Path = Path("./input_paths.txt"),
    output_paths_file: Path = Path("./output_paths.txt")
) -> None:
    """Handle the tidy-graph command logic."""

    valid_input_paths = validate_and_clean_paths(input_paths_file)
    valid_output_paths = validate_and_clean_paths(output_paths_file)

    # Select input graph
    graph_path = select_input_graph(valid_input_paths)
    save_path_to_file(graph_path, input_paths_file)

    # Extract and save tags
    typer.echo("Extracting unique #hashtags and [[backlinks]]...")
    tags = extract_tags(graph_path)
    tags_file = graph_path / "tags.txt"
    save_tags_to_file(tags, tags_file)
    typer.echo("Extracted tags have been saved to {}".format(tags_file))

    # Collect content and estimate costs
    content_list = collect_content_list(graph_path)
    total_files = len(content_list)
    sample_size = min(3, total_files)

    # Calculate different cost scenarios
    standard_cost = estimate_cost(content_list, model, use_batch=False)
    batch_cost = estimate_cost(content_list, model, use_batch=True)
    cached_cost = estimate_cost(
        content_list, model, use_batch=True, assume_cached=True)
    pricing = load_pricing()

    display_cost_estimates(standard_cost, batch_cost, cached_cost,
                           total_files, sample_size, model, pricing)

    # Handle sample processing
    if typer.confirm("\nWould you like to process a small sample first?"):
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            if not loop.run_until_complete(process_sample(content_list, sample_size, tags, model)):
                typer.echo("Operation cancelled.")
                raise typer.Exit()
        finally:
            loop.close()
    elif not typer.confirm("\nWould you like to proceed with processing the entire graph?"):
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    # Generate output path
    output_name = generate_output_name(graph_path)
    output_base = graph_path.parent / output_name
    output_path = output_base
    counter = 1

    while output_path.exists():
        output_path = output_base.parent / \
            "{}_{}".format(output_base.name, counter)
        counter += 1

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    save_path_to_file(output_path, output_paths_file)

    # Process files
    typer.echo("\nProcessing the graph...")
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(process_files(
            content_list, output_path, tags, model))
    finally:
        loop.close()

    # Copy assets
    if copy_assets_folder(graph_path, output_path):
        typer.echo("Assets folder copied successfully")

    typer.echo("Tidied graph has been saved to {}".format(output_path))


def select_input_graph(valid_input_paths: List[Path]) -> Path:
    """Select input graph path either from history or by browsing."""
    if valid_input_paths:
        use_existing = questionary.confirm(
            "Would you like to select from previously used graphs?",
            default=True
        ).ask()

        if use_existing:
            choices = [str(p) for p in valid_input_paths] + \
                ["[Browse for new location]"]
            selected = questionary.select(
                "Select input graph location:",
                choices=choices
            ).ask()

            if selected != "[Browse for new location]":
                return Path(selected)

    typer.echo("Please select the Logseq graph directory to process...")
    return select_path_interactively()


async def process_files(content_list: List[Tuple[str, Path]], output_path: Path,
                        tags: set, model: str) -> None:
    """Process all files with rate limiting."""
    rate_limiter = RateLimiter(500, 150000)  # RPM and TPM limits
    await process_files_with_progress(
        content_list=content_list,
        output_path=output_path,
        tags=tags,
        model=model,
        rate_limiter=rate_limiter
    )
