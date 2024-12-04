"""Prompts for the Logseq tidy graph tool."""

TIDY_PROMPT = """You are tasked with rewriting and organizing Logseq page content into a clear, concise format.

Key requirements:
- Determine the content type (daily journal, meeting notes, project overview, decision record, etc.) and format appropriately.
- Use bullet points extensively for clarity.
- Keep summaries concise and factual.
- Remove conversational elements and personal commentary.
- Preserve all:
  - [[backlinks]] and #hashtags; keep them in-line with relevant text.
  - Web links (on their own line).
  - Asset links and images ![image.png](../assets/image.png) (on their own line).
- Organize the content into a hierarchical tree of points:
  - Group related items together under relevant higher-level topics.
  - Add new higher-level blocks where appropriate, using relevant [[backlinks]] or #tags as headings.
  - Indent child points under their parent bullet points to create a clear structure.
  - For flat inputs, transform them by adding relevant parent topics as parent bullets with the child points indented underneath.
- Include inline references with appropriate [[backlinks]] that are likely related.
- **Replace mentions of people's names with their [[backlink]] if their name exists in the list of tags/backlinks.**
- Add a #Review tag if content needs human attention.

For daily journals/logs:
- Group by activity type (meetings, tasks, decisions, notes).
- Use timestamps if present.
- Convert stream-of-consciousness into structured points.

For technical content/decisions:
- Include clear context and rationale.
- Highlight key decisions and outcomes.
- List implications or next steps.

Use these extracted tags and backlinks where relevant:
{}

Content to rewrite:
{}"""
