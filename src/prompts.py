"""Prompts for the Logseq tidy graph tool."""

TIDY_PROMPT = """You are tasked with rewriting and organizing Logseq page content into a clear, concise format while creating meaningful connections between related pages and topics.

Key requirements:
- Determine the content type (daily journal, meeting notes, project overview, decision record, etc.) and format appropriately.
- Use bullet points extensively for clarity.
- Keep summaries concise and factual.
- Remove conversational elements and personal commentary.


Content Organization and Linking:
- Preserve and enhance all connections:
  - [[backlinks]] and #hashtags inline with relevant text
  - Web links (on their own line)
  - Asset links and images ![image.png](../assets/image.png) (on their own line)
- Create intelligent links to related pages:
  - Link to overview/MOC pages when discussing broad topics
  - Reference project pages when mentioning specific projects
  - Link to query pages that collect related information
  - Connect to topic pages that provide context or details
- **Replace mentions of people, projects, and topics with their [[backlink]] if they exist in the available context**

Hierarchical Structure:
- Organize content into a clear tree structure:
  - Group related items under meaningful parent topics
  - Use relevant [[backlinks]] or #tags as section headings
  - Indent child points for clear relationships
  - Transform flat content by adding contextual parent topics
- Start major sections with links to relevant overview pages
- Reference appropriate query pages that aggregate related content

Content-Type Specific Guidelines:
For daily journals/logs:
- Group by activity type (meetings, tasks, decisions, notes)
- Link each activity to relevant project and topic pages
- Use timestamps if present
- Convert stream-of-consciousness into structured points
- Reference query pages that track similar activities

For technical content/decisions:
- Include clear context and rationale
- Link to related technical documentation and overview pages
- Highlight key decisions and outcomes
- List implications or next steps
- Connect to query pages tracking similar decisions

For project updates:
- Link to the main project page
- Reference related overview/MOC pages
- Connect to relevant topic pages
- Link to query pages tracking project items

Add #Review tag if content needs human attention.

Available Context:
Tags and Backlinks:
{}

Pages Information (with types and relationships):
{}

Content to rewrite:
{}"""
