import json
from pathlib import Path
from typing import Dict, List, Tuple
import tiktoken
import aiohttp
from rich.console import Console
import asyncio
from asyncio import Semaphore
from datetime import datetime, timedelta

CONFIG_DIR = Path("config")
PRICING_FILE = CONFIG_DIR / "model_pricing.json"

# Rate limit settings
RPM_LIMIT = 500  # Requests per minute
TPM_LIMIT = 150000  # Tokens per minute
MAX_PARALLEL_REQUESTS = 15  # Maximum parallel requests


class RateLimiter:
    def __init__(self, rpm_limit: int, tpm_limit: int):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests = []
        self.tokens = []
        self.semaphore = Semaphore(MAX_PARALLEL_REQUESTS)

    async def acquire(self, tokens: int):
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean up old entries
        self.requests = [t for t in self.requests if t > minute_ago]
        self.tokens = [(t, tok) for t, tok in self.tokens if t > minute_ago]

        # Check rate limits
        while len(self.requests) >= self.rpm_limit or sum(tok for _, tok in self.tokens) >= self.tpm_limit:
            await asyncio.sleep(0.1)
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            self.requests = [t for t in self.requests if t > minute_ago]
            self.tokens = [(t, tok)
                           for t, tok in self.tokens if t > minute_ago]

        # Acquire semaphore for parallel request limiting
        await self.semaphore.acquire()

        # Record this request
        self.requests.append(now)
        self.tokens.append((now, tokens))

    def release(self):
        self.semaphore.release()


async def fetch_model_pricing() -> Dict:
    """Fetch current model pricing from OpenAI API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.openai.com/v1/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    return {}
    except Exception as e:
        console = Console()
        console.print(
            "[yellow]Warning: Could not fetch model pricing: {}".format(e))
    return {}


def load_pricing() -> Dict:
    """Load pricing from local config file, creating default if needed."""
    if not CONFIG_DIR.exists():
        CONFIG_DIR.mkdir(parents=True)

    if not PRICING_FILE.exists():
        default_pricing = {
            "gpt-4o": {
                "input": 2.50,
                "input_batch": 1.25,
                "input_cached": 1.25,
                "output": 10.00,
                "output_batch": 5.00
            },
            "gpt-4o-mini": {
                "input": 0.150,
                "input_batch": 0.075,
                "input_cached": 0.075,
                "output": 0.600,
                "output_batch": 0.300
            }
        }
        PRICING_FILE.write_text(json.dumps(default_pricing, indent=2))
        return default_pricing

    try:
        return json.loads(PRICING_FILE.read_text())
    except Exception as e:
        console = Console()
        console.print(
            "[yellow]Warning: Error loading pricing file: {}".format(e))
        return {}


def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_cost(
    content_list: List[Tuple[str, Path]],
    model: str = "gpt-4o-mini",
    avg_output_tokens: int = 300,
    use_batch: bool = True,
    assume_cached: bool = False
) -> dict:
    """
    Estimate processing costs with detailed breakdown.
    Returns a dictionary with cost details.
    """
    pricing = load_pricing()

    if model not in pricing:
        raise ValueError(
            "Pricing information for model {} is not available. Please update {} with current pricing.".format(
                model, PRICING_FILE)
        )

    model_pricing = pricing[model]

    if assume_cached and "input_cached" in model_pricing:
        input_token_cost = model_pricing["input_cached"] / 1_000_000
    elif use_batch and "input_batch" in model_pricing:
        input_token_cost = model_pricing["input_batch"] / 1_000_000
    else:
        input_token_cost = model_pricing["input"] / 1_000_000

    output_token_cost = (
        model_pricing["output_batch"] / 1_000_000 if use_batch and "output_batch" in model_pricing
        else model_pricing["output"] / 1_000_000
    )

    total_input_tokens = sum(count_tokens(content, model)
                             for content, _ in content_list)
    total_output_tokens = len(content_list) * avg_output_tokens

    input_cost = total_input_tokens * input_token_cost
    output_cost = total_output_tokens * output_token_cost
    total_cost = input_cost + output_cost

    return {
        "total_cost": total_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "pricing_type": (
            "cached" if assume_cached else
            "batch" if use_batch else
            "standard"
        )
    }
