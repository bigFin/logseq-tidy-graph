import sys
import asyncio
from typing import Optional
import typer
from src.command_handler import handle_tidy_graph_command, DEFAULT_MODEL

app = typer.Typer(no_args_is_help=True)


@app.command(name="tidy-graph", help="Process and tidy a Logseq graph")
def tidy_graph(
    model: str = typer.Option(
        DEFAULT_MODEL, help="OpenAI model to use (e.g., gpt-4o-mini, gpt-3.5-turbo)"),
    update_pricing: bool = typer.Option(
        False, "--update-pricing", help="Update model pricing information")
) -> None:
    """Process and tidy a Logseq graph."""
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy())

        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run the async command
        loop.run_until_complete(
            handle_tidy_graph_command(model, update_pricing))

    except KeyboardInterrupt:
        typer.echo("\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"\nError: {str(e)}", err=True)
        raise typer.Exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    app()
