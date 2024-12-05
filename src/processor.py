from pathlib import Path
import re
from typing import Set, Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PageInfo:
    """Store information about a page."""
    title: str
    content: str
    tags: Set[str]
    backlinks: Set[str]
    is_query: bool = False
    is_overview: bool = False


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


def extract_pages(graph_path: Path) -> Dict[str, PageInfo]:
    """Extract all pages and their metadata from the graph."""
    pages = {}

    for dir_name in ['journals', 'pages']:
        dir_path = graph_path / dir_name
        if not dir_path.exists():
            continue

        for file_path in dir_path.glob("*.md"):
            content = file_path.read_text()
            title = file_path.stem

            # Extract tags and backlinks
            hashtags = set(re.findall(
                r'#([^\s#]+(?:\[\[.*?\]\])?[^\s#]*)', content))
            backlinks = set(re.findall(r'\[\[(.*?)\]\]', content))

            # Detect if page is a query or overview
            is_query = any(marker in content.lower() for marker in [
                'query-table::',
                'query-list::',
                'query-properties::'
            ])

            is_overview = any(marker in title.lower() for marker in [
                'overview',
                'summary',
                'index',
                'moc'  # Map of Content
            ]) or len(backlinks) > 10  # Pages with many backlinks are likely overviews

            pages[title] = PageInfo(
                title=title,
                content=content,
                tags=hashtags,
                backlinks=backlinks,
                is_query=is_query,
                is_overview=is_overview
            )

    return pages


def save_pages_info(pages: Dict[str, PageInfo], output_dir: Path) -> None:
    """Save pages information to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all pages info
    pages_info = []
    for page in pages.values():
        info = {
            'title': page.title,
            'tags': list(page.tags),
            'backlinks': list(page.backlinks),
            'is_query': page.is_query,
            'is_overview': page.is_overview
        }
        pages_info.append(info)

    # Save special pages lists
    query_pages = [p.title for p in pages.values() if p.is_query]
    overview_pages = [p.title for p in pages.values() if p.is_overview]

    # Save to files
    (output_dir / 'pages_info.txt').write_text(
        '\n'.join(f"{p['title']}: {len(p['backlinks'])} backlinks, {len(p['tags'])} tags"
                  for p in pages_info)
    )
    (output_dir / 'query_pages.txt').write_text('\n'.join(sorted(query_pages)))
    (output_dir / 'overview_pages.txt').write_text('\n'.join(sorted(overview_pages)))


def process_graph(content: str, context: Dict[str, PageInfo]) -> str:
    """Process a single graph file with knowledge of other pages."""
    # This is a placeholder - actual implementation would depend on your needs
    return content
