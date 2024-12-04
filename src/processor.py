import os
import re
from pathlib import Path
from typing import Set
import openai
from src.tidy import tidy_content


def extract_tags(graph_path: Path) -> Set[str]:
    """
    Extract unique #hashtags and [[backlinks]] from all .md files in the graph.
    """
    journals_dir = graph_path / "journals"
    pages_dir = graph_path / "pages"

    tags = set()

    # Regex patterns
    hashtag_pattern = re.compile(r"#\w+")
    backlink_pattern = re.compile(r"\[\[.+?\]\]")

    # Process all .md files in journals and pages
    for directory in [journals_dir, pages_dir]:
        for file in directory.glob("*.md"):
            with file.open("r") as f:
                content = f.read()
                tags.update(hashtag_pattern.findall(content))
                tags.update(backlink_pattern.findall(content))

    return tags


def save_tags_to_file(tags: Set[str], output_file: Path):
    """
    Save extracted tags to a file.
    """
    with output_file.open("w") as f:
        for tag in sorted(tags):
            f.write(f"{tag}\n")


def estimate_cost(graph_path: Path, tokens_per_file: int = 500) -> float:
    """
    Estimate the cost of processing the graph based on the number of files and average tokens per file.
    """
    journals_dir = graph_path / "journals"
    pages_dir = graph_path / "pages"

    # Count the number of files in journals and pages
    journal_files = list(journals_dir.glob("*.md"))
    page_files = list(pages_dir.glob("*.md"))
    total_files = len(journal_files) + len(page_files)

    # Calculate total tokens
    total_tokens = total_files * tokens_per_file

    # OpenAI pricing: $0.03 per 1K input tokens, $0.06 per 1K output tokens
    cost_per_1k_tokens_input = 0.03
    cost_per_1k_tokens_output = 0.06
    cost = (total_tokens / 1000) * \
        (cost_per_1k_tokens_input + cost_per_1k_tokens_output)

    return cost


def process_graph(graph_path: Path, output_path: Path, tags: Set[str]):
    """
    Process the Logseq graph to tidy its contents.
    """
    journals_dir = graph_path / "journals"
    pages_dir = graph_path / "pages"
    output_journals_dir = output_path / "journals"
    output_pages_dir = output_path / "pages"

    output_journals_dir.mkdir(parents=True, exist_ok=True)
    output_pages_dir.mkdir(parents=True, exist_ok=True)

    # Process journals
    for file in journals_dir.glob("*.md"):
        process_file(file, output_journals_dir, tags)

    # Process pages
    for file in pages_dir.glob("*.md"):
        process_file(file, output_pages_dir, tags)


def process_file(file_path: Path, output_dir: Path, tags: Set[str]):
    """
    Send file content to OpenAI and save the tidied version.
    """
    with file_path.open("r") as file:
        content = file.read()

    # Use the OpenAI API to tidy content
    tidied_content = tidy_content(content, tags)

    # Save the tidied content
    output_file = output_dir / file_path.name
    with output_file.open("w") as file:
        file.write(tidied_content)
