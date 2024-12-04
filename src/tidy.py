import openai
from typing import Set, List
import re


def tidy_content(content: str, tags: Set[str], model: str = "gpt-4o-mini") -> str:
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
    prompt = f"""
You are tasked with rewriting a Logseq page into a professional Architectural Decision Record (ADR) style. 
- Organize the content into sections where possible, including:
  - Title: A concise summary of the decision.
  - Context: Background and key details about the problem or decision.
  - Decision: The specific choice or action taken.
  - Consequences: Any impacts, benefits, or trade-offs resulting from the decision.
- Preserve the following exactly as written:
  - Queries: Lines starting with {{query ...}} or query-properties:: must remain unaltered.
  - Page Properties: Lines like type::, status::, and bucket:: must not be changed.
- Preserve and improve [[backlinks]] and #hashtags.
- Rewrite informal language or personal commentary into a professional tone suitable for sharing.
- Add a #Review tag for content that may still need human attention.
- Use these extracted tags and backlinks where relevant:
{tags_context}

Content to rewrite:
{content_without_properties}
"""

    try:
        # Send the prompt to OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
        )
        tidied_content = response["choices"][0]["message"]["content"]

        # Reinsert queries and properties into the tidied content
        result = "\n".join(properties) + "\n" + \
            tidied_content + "\n" + "\n".join(queries)
        return result

    except Exception as e:
        raise RuntimeError(f"Error in OpenAI API call: {e}")
