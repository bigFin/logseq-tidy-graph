from pathlib import Path
import re
from typing import Set


def extract_tags(graph_path: Path) -> Set[str]:
    """Extract all unique hashtags and backlinks from the graph."""
    tags = set()

    # Process both journals and pages directories
    for dir_name in ['journals', 'pages']:
        dir_path = graph_path / dir_name
        if not dir_path.exists():
            continue

        for file_path in dir_path.glob("*.md"):
            content = file_path.read_text()

            # Extract hashtags
            hashtags = re.findall(
                r'#([^\s#]+(?:\[\[.*?\]\])?[^\s#]*)', content)
            tags.update(hashtags)

            # Extract backlinks
            backlinks = re.findall(r'\[\[(.*?)\]\]', content)
            tags.update(backlinks)

    return tags


def save_tags_to_file(tags: Set[str], tags_file: Path) -> None:
    """Save extracted tags to a file."""
    tags_file.write_text('\n'.join(sorted(tags)))


def process_graph(content: str) -> str:
    """Process a single graph file."""
    # This is a placeholder - actual implementation would depend on your needs
    return content
