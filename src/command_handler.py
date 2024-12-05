import sys
from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import asyncio
import typer
import questionary
from rich.console import Console

from .ui_helper import (
    save_path_to_file,
    generate_output_name,
    select_path_interactively,
    display_cost_estimates
)
from .file_processor import (
    process_files_with_progress,
    collect_content_list,
    copy_assets_folder
)
from .cost_manager import (
    RateLimiter,
    fetch_model_pricing,
    load_pricing,
    estimate_cost
)
from .processor import (
    process_graph,
    extract_tags,
    save_tags_to_file,
    extract_pages,
    save_pages_info,
    PageInfo
)

from .validator import validate_and_clean_paths

from .cost_manager import RPM_LIMIT, TPM_LIMIT, MAX_PARALLEL_REQUESTS

DEFAULT_MODEL = "gpt-4o-mini"

SAMPLE_SIZE = 3


@dataclass
class ProcessingContext:
    """Contains all necessary context for processing a graph."""
    graph_path: Path
    output_path: Path
    content_list: List[Tuple[str, Path]]
    tags: set
    pages: Dict[str, PageInfo]
    model: str
    rate_limiter: RateLimiter


def get_sample_content(
    content_list: List[Tuple[str, Path]],
    sample_size: int
) -> List[Tuple[str, Path]]:
    """Get a balanced sample of pages and journals."""
    pages_sample = [c for c in content_list if 'pages' in str(
        c[1].parent)][:sample_size//2]
    journals_sample = [c for c in content_list if 'journals' in str(
        c[1].parent)][:sample_size//2]

    if not pages_sample and not journals_sample:
        return content_list[:sample_size]

    sample_content = []
    sample_content.extend(pages_sample)
    remaining_slots = sample_size - len(sample_content)
    if remaining_slots > 0:
        sample_content.extend(journals_sample[:remaining_slots])

    return sample_content if sample_content else content_list[:sample_size]


def create_processing_context(
    graph_path: Path,
    model: str,
    output_path: Optional[Path] = None
) -> ProcessingContext:
    """Create a processing context with all necessary data."""
    tags = extract_tags(graph_path)
    pages = extract_pages(graph_path)
    content_list = collect_content_list(graph_path)

    # Create rate limiter with default settings from cost_manager
    rate_limiter = RateLimiter(
        rpm_limit=RPM_LIMIT,
        tpm_limit=TPM_LIMIT,
        max_parallel=MAX_PARALLEL_REQUESTS
    )

    return ProcessingContext(
        graph_path=graph_path,
        output_path=output_path or graph_path,
        content_list=content_list,
        tags=tags,
        pages=pages,
        model=model,
        rate_limiter=rate_limiter
    )


def save_metadata(ctx: ProcessingContext) -> Path:
    """Save metadata (tags and pages info) to disk."""
    metadata_dir = ctx.graph_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    # Save tags
    tags_file = metadata_dir / "tags.txt"
    save_tags_to_file(ctx.tags, tags_file)

    # Save pages info
    save_pages_info(ctx.pages, metadata_dir)

    return metadata_dir


async def process_sample(
    ctx: ProcessingContext,
    sample_size: int
) -> bool:
    """Process a sample of files and return whether to continue."""
    temp_output = Path("./sample_output")
    temp_output.mkdir(exist_ok=True)

    sample_content = get_sample_content(ctx.content_list, sample_size)

    typer.echo("\nProcessing sample files...")

    # Convert content list to include pages context
    content_with_pages = [(content, path) for content, path in sample_content]

    await process_files_with_progress(
        content_list=content_with_pages,
        output_path=temp_output,
        tags=ctx.tags,
        model=ctx.model,
        rate_limiter=ctx.rate_limiter,
        pages=ctx.pages
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


async def process_full_graph(ctx: ProcessingContext) -> None:
    """Process the entire graph."""
    typer.echo("\nProcessing the graph...")
    # Convert content list to include pages context
    content_with_pages = [(content, path)
                          for content, path in ctx.content_list]

    await process_files_with_progress(
        content_list=content_with_pages,
        output_path=ctx.output_path,
        tags=ctx.tags,
        model=ctx.model,
        rate_limiter=ctx.rate_limiter,
        pages=ctx.pages
    )

    # Copy assets
    if copy_assets_folder(ctx.graph_path, ctx.output_path):
        typer.echo("Assets folder copied successfully")

    typer.echo("Tidied graph has been saved to {}".format(ctx.output_path))


def generate_output_path(graph_path: Path) -> Path:
    """Generate a unique output path for the processed graph."""
    output_name = generate_output_name(graph_path)
    output_base = graph_path.parent / output_name
    output_path = output_base
    counter = 1

    while output_path.exists():
        output_path = output_base.parent / \
            "{}_{}".format(output_base.name, counter)
        counter += 1

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def display_metadata_stats(ctx: ProcessingContext, metadata_dir: Path) -> None:
    """Display statistics about the extracted metadata."""
    typer.echo("Metadata extracted and saved to {}/".format(metadata_dir))
    typer.echo("Found {} unique tags and {} pages".format(
        len(ctx.tags), len(ctx.pages)))
    typer.echo(
        "Including {} queries and {} overview pages".format(
            sum(1 for p in ctx.pages.values() if p.is_query),
            sum(1 for p in ctx.pages.values() if p.is_overview)
        )
    )


async def select_input_graph(valid_input_paths: List[Path]) -> Path:
    """Select input graph path either from history or by browsing."""
    if valid_input_paths:
        use_existing = await questionary.confirm(
            "Would you like to select from previously used graphs?",
            default=True
        ).ask_async()

        if use_existing:
            choices = [str(p) for p in valid_input_paths] + \
                ["[Browse for new location]"]
            selected = await questionary.select(
                "Select input graph location:",
                choices=choices
            ).ask_async()

            if selected != "[Browse for new location]":
                return Path(selected)

    typer.echo("Please select the Logseq graph directory to process...")
    return await select_path_interactively()


async def handle_tidy_graph_command(
    model: str = DEFAULT_MODEL,
    update_pricing: bool = False,
    input_paths_file: Path = Path("./input_paths.txt"),
    output_paths_file: Path = Path("./output_paths.txt")
) -> None:
    """Implementation of the tidy-graph command logic."""

    # Update pricing if requested
    if update_pricing:
        await fetch_model_pricing()
        typer.echo("Model pricing information updated.")

    # Initialize paths and context
    valid_input_paths = validate_and_clean_paths(input_paths_file)

    graph_path = await select_input_graph(valid_input_paths)
    save_path_to_file(graph_path, input_paths_file)

    # Create processing context
    ctx = create_processing_context(graph_path, model)

    # Save and display metadata
    metadata_dir = save_metadata(ctx)
    display_metadata_stats(ctx, metadata_dir)

    # Display cost estimates
    total_files = len(ctx.content_list)
    sample_size = min(SAMPLE_SIZE, total_files)

    standard_cost = estimate_cost(ctx.content_list, ctx.model, use_batch=False)
    batch_cost = estimate_cost(ctx.content_list, ctx.model, use_batch=True)
    cached_cost = estimate_cost(
        ctx.content_list, ctx.model, use_batch=True, assume_cached=True)
    pricing = load_pricing()

    display_cost_estimates(
        standard_cost, batch_cost, cached_cost,
        total_files, sample_size, ctx.model, pricing
    )

    # Handle sample processing
    if await questionary.confirm("\nWould you like to process a small sample first?").ask_async():
        if not await process_sample(ctx, sample_size):
            typer.echo("Operation cancelled.")
            raise typer.Exit()
    elif not await questionary.confirm("\nWould you like to proceed with processing the entire graph?").ask_async():
        typer.echo("Operation cancelled.")
        raise typer.Exit()

    # Setup output path
    output_path = generate_output_path(graph_path)
    save_path_to_file(output_path, output_paths_file)

    # Update context with output path
    ctx = replace(ctx, output_path=output_path)

    # Process the full graph
    await process_full_graph(ctx)
