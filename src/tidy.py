from openai import AsyncOpenAI
from typing import Set, List, Dict, Tuple, Generator, AsyncGenerator, Optional
from pathlib import Path
from .processor import PageInfo
import re
import asyncio
from itertools import islice
from .prompts import TIDY_PROMPT

# Initialize the AsyncOpenAI client
client = AsyncOpenAI()


async def tidy_content_batch_stream(
    contents: List[Tuple[str, Set[str], Dict[str, PageInfo]]],
    model: str = "gpt-4o-mini",
    batch_size: int = 20
) -> AsyncGenerator[str, None]:
    """
    Batch process multiple Logseq page contents using OpenAI GPT model.

    Args:
        contents (List[Tuple[str, Set[str]]]): List of tuples containing (content, tags) pairs to process.
        model (str): OpenAI model to use (default: gpt-4o-mini).

    Returns:
        List[str]: List of tidied contents in the same order as input.
    """
    async def process_batch(batch_contents: List[Tuple[str, Set[str]]]) -> List[str]:
        prepared_contents = []
        preserved_elements = []

        for content, tags, pages in batch_contents:
            # Extract and preserve queries and page properties using regex
            query_pattern = re.compile(
                r"query-properties::.*|{{query.*}}", re.MULTILINE)
            property_pattern = re.compile(r"^\w+::.*$", re.MULTILINE)

            queries = query_pattern.findall(content)
            properties = property_pattern.findall(content)

            # Remove queries and properties from the content
            content_without_queries = query_pattern.sub("", content)
            content_without_properties = property_pattern.sub(
                "", content_without_queries)

            # Prepare prompt for each content
            tags_context = "\n".join(tags)

            # Format pages information
            pages_context = []
            for page_name, page_info in pages.items():
                page_type = []
                if page_info.is_overview:
                    page_type.append("Overview/MOC")
                if page_info.is_query:
                    page_type.append("Query")
                if page_info.tags:
                    page_type.append(f"Tags: {', '.join(page_info.tags)}")

                page_desc = f"[[{page_name}]] - {' | '.join(page_type)}"
                pages_context.append(page_desc)

            pages_context_str = "\n".join(pages_context)

            prompt = TIDY_PROMPT.format(
                tags_context,
                pages_context_str,
                content_without_properties
            )
            prepared_contents.append({"role": "system", "content": prompt})
            preserved_elements.append((properties, queries))

        try:
            # Send batch request to OpenAI
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": content["content"]}
                          for content in prepared_contents]
            )

            # Process results
            results = []
            for idx, choice in enumerate(response.choices):
                properties, queries = preserved_elements[idx]
                tidied_content = choice.message.content

                # Reinsert queries and properties
                result = "\n".join(properties) + "\n" + \
                    tidied_content + "\n" + "\n".join(queries)
                results.append(result)

            return results

        except Exception as e:
            raise RuntimeError("Error in OpenAI API batch call: {}".format(e))

    # Process content in batches
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i + batch_size]
        results = await process_batch(batch)
        for result in results:
            yield result


async def tidy_content(
    content: str,
    tags: Set[str],
    pages: Dict[str, PageInfo],
    model: str = "gpt-4o-mini"
) -> str:
    """
    Use OpenAI GPT-4o-mini (or specified model) to tidy up Logseq page content, including relevant tags, 
    while preserving queries and page properties.

    Args:
        content (str): The text content to process.
        tags (Set[str]): A set of extracted hashtags and backlinks to include in the prompt.
        model (str): OpenAI model to use (default: gpt-4o-mini).

    Returns:
        str: The tidied content.
    """
    # Prepare the tags as context for the prompt
    tags_context = "\n".join(tags)

    # Extract and preserve queries and page properties using regex
    query_pattern = re.compile(
        r"query-properties::.*|{{query.*}}", re.MULTILINE)
    property_pattern = re.compile(r"^\w+::.*$", re.MULTILINE)

    queries = query_pattern.findall(content)
    properties = property_pattern.findall(content)

    # Remove queries and properties from the content to avoid rewriting them
    content_without_queries = query_pattern.sub("", content)
    content_without_properties = property_pattern.sub(
        "", content_without_queries)

    # Formulate the prompt
    prompt = TIDY_PROMPT.format(tags_context, content_without_properties)

    # Use the batch function with a single item
    async for result in tidy_content_batch_stream([(content, tags, pages)], model=model, batch_size=1):
        return result
