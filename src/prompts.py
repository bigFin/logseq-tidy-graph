"""Prompts for the Logseq tidy graph tool."""

TIDY_PROMPT = """You are tasked with rewriting and organizing Logseq page content into a clear, concise format.

Key requirements:
- Determine the content type (daily journal, meeting notes, project overview, decision record, etc.) and format appropriately
- Use bullet points extensively for clarity
- Keep summaries concise and factual
- Remove conversational elements and personal commentary
- Group related items together under headers
- Preserve all:
  - [[backlinks]] and #hashtags
  - Web links (on their own line)
  - Asset links and images ![image.png](../assets/image.png) (on their own line)
- Add a #Review tag if content needs human attention

For daily journals/logs:
- Group by activity type (meetings, tasks, decisions, notes)
- Use timestamps if present
- Convert stream-of-consciousness into structured points

For technical content/decisions:
- Include clear context and rationale
- Highlight key decisions and outcomes
- List implications or next steps

Use these extracted tags and backlinks where relevant:
{}

Content to rewrite:
{}"""
