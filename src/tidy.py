import openai
from typing import Set


def tidy_content(content: str, tags: Set[str]) -> str:
    """
    Use OpenAI GPT-4 to tidy up Logseq page content, including relevant tags.
    """
    # Prepare the tags as context for the prompt
    tags_context = "\n".join(tags)

    # Formulate the prompt
    prompt = f"""
You are tasked with cleaning up a Logseq page for professional use. 
- Summarize the content while keeping and improving [[backlinks]] and #hashtags.
- Rewrite personal commentary into professional language.
- Add #Review for content that may be unprofessional.
- Use the following tags when relevant:
{tags_context}

Content to tidy:
{content}
"""
    try:
        # Send the prompt to OpenAI GPT-4
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Error in OpenAI API call: {e}")
