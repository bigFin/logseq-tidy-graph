from pathlib import Path
from typing import List, Tuple, Set
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
import asyncio
from src.cost_manager import RateLimiter
from src import tidy
import typer


async def process_single_file(
    content: str,
    file_path: Path,
    output_path: Path,
    tags: set,
    model: str,
    progress,
    task_id,
    console: Console,
    rate_limiter: RateLimiter,
    pages: dict = None
) -> bool:
    """Process a single file and return True if successful. Raises exceptions for retry-able errors."""
    """Process a single file and return True if successful."""
    try:
        progress.update(
            task_id, description="Processing: {}".format(file_path.name))

        # Estimate tokens (rough estimate)
        estimated_tokens = len(content.split()) * 2  # rough estimate of tokens

        # Acquire rate limit
        await rate_limiter.acquire(estimated_tokens)
        try:
            async for result in tidy.tidy_content_batch_stream([(content, tags, pages or {})], model=model, batch_size=1):
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
        finally:
            rate_limiter.release()

        raise RuntimeError("No result received for {}".format(file_path.name))

    except Exception as e:
        console.print("[red]Error processing {}: {}".format(file_path, str(e)))
        progress.update(task_id, advance=1)
        return False


async def process_files_with_progress(
    content_list: List[Tuple[str, Path]],
    output_path: Path,
    tags: Set[str],
    model: str,
    rate_limiter: RateLimiter,
    pages: dict = None
) -> List[Tuple[str, Path]]:
    """Process files with progress monitoring. Returns list of failed files with their content."""
    if pages is None:
        pages = {}

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
                # Create a task for processing each file
                task = process_single_file(
                    content=content,
                    file_path=file_path,
                    output_path=output_path,
                    tags=tags,
                    model=model,
                    progress=progress,
                    task_id=file_progress,
                    console=console,
                    rate_limiter=rate_limiter,
                    pages=pages
                )
                tasks.append((task, file_path))

            # Process batch with rate limit handling
            results = await asyncio.gather(*(task for task, _ in tasks), return_exceptions=True)

            for (_, file_path), result in zip(tasks, results):
                content = next(c for c, p in content_list if p == file_path)
                if isinstance(result, Exception):
                    error_msg = str(result)
                    if "rate_limit_exceeded" in error_msg:
                        # Extract wait time from error message if available
                        import re
                        wait_time = 2.0  # default wait time
                        match = re.search(
                            r'try again in (\d+\.?\d*)s', error_msg)
                        if match:
                            wait_time = float(match.group(1))
                        console.print(
                            f"[yellow]Rate limit hit for {file_path}. Will retry later.")
                        await asyncio.sleep(wait_time)
                        failed_files.append((content, file_path))
                    else:
                        console.print(
                            f"[red]Error processing {file_path}: {error_msg}")
                        failed_files.append((content, file_path))
                elif result:
                    successful_files += 1
                else:
                    failed_files.append((content, file_path))

                progress.update(overall_progress, advance=1)

            if len(failed_files) > 0 and (i + batch_size) < len(content_list):
                if not typer.confirm("\nContinue processing remaining files?", default=True):
                    # Return both processed and unprocessed failed files
                    remaining = [(c, p)
                                 for c, p in content_list[i+batch_size:]]
                    return failed_files + remaining

    return failed_files


def collect_content_list(graph_path: Path) -> List[Tuple[str, Path]]:
    """Collect all markdown files from journals and pages directories."""
    content_list = []
    for dir_name in ['journals', 'pages']:
        dir_path = graph_path / dir_name
        if dir_path.exists():
            for file_path in dir_path.glob("*.md"):
                content_list.append((file_path.read_text(), file_path))
    return content_list


def copy_assets_folder(src_path: Path, dst_path: Path) -> bool:
    """Copy assets folder from source to destination."""
    src_assets = src_path / "assets"
    if src_assets.exists():
        from shutil import copytree
        dst_assets = dst_path / "assets"
        try:
            copytree(src_assets, dst_assets, dirs_exist_ok=True)
            return True
        except Exception:
            return False
    return False
