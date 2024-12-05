# Logseq Tidy Graph

A CLI tool to professionalize and enhance Logseq graph contents using AI.

    ## Features

- **Smart Graph Processing**
  - Interactive filesystem navigation for graph selection
  - Preserves #hashtags and [[page references]]
  - Maintains page relationships and knowledge structure
  - Processes both journal entries and pages

- **AI Enhancement**
  - Rewrites content in professional tone using GPT-4
  - Context-aware processing using page references
  - Sample processing preview before full commit
  - Rate-limited API usage to stay within OpenAI limits

- **Safety & Control**
  - Non-destructive: saves to new output directory
  - Preserves original graph structure
  - Estimates OpenAI API cost before processing
  - Progress tracking with error handling

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the tool:
```bash
python cli.py tidy-graph
```

3. Follow the interactive prompts to:
   - Select your Logseq graph
   - Review sample processing
   - Confirm full graph processing

## Rate Limits

The tool automatically respects OpenAI's rate limits:
- Tokens per minute (TPM): 200,000
- Concurrent requests: Automatically managed
- Built-in rate limiting and request queuing

## Output

Processed files maintain the same structure as your original graph while providing:
- Professional tone and clarity
- Preserved links and references
- Original Logseq formatting
- Separate output directory for safety

