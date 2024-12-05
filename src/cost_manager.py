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
AVG_TOKENS_PER_REQUEST = 4000  # Conservative estimate for input + output tokens
TPM_LIMIT = 200000  # OpenAI's token per minute limit
# Calculated from TPM
RPM_LIMIT = min(100, TPM_LIMIT // AVG_TOKENS_PER_REQUEST)
# Conservative parallel limit
MAX_PARALLEL_REQUESTS = min(5, TPM_LIMIT // (AVG_TOKENS_PER_REQUEST * 4))


class RateLimiter:
    def __init__(self, rpm_limit: int, tpm_limit: int, max_parallel: int = MAX_PARALLEL_REQUESTS):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

        # Validate settings
        if max_parallel * AVG_TOKENS_PER_REQUEST > tpm_limit:
            raise ValueError(
                f"max_parallel setting would exceed TPM limit: {max_parallel} * {AVG_TOKENS_PER_REQUEST} > {tpm_limit}")

        self.requests = []
        self.tokens = []
        self.current_token_sum = 0
        self.semaphore = Semaphore(max_parallel)
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int):
        async with self.lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean up old entries and recalculate token sum
            self.requests = [t for t in self.requests if t > minute_ago]
            self.tokens = [(t, tok)
                           for t, tok in self.tokens if t > minute_ago]
            self.current_token_sum = sum(tok for _, tok in self.tokens)

        # Predictive check - will this request put us over the limit?
        while (len(self.requests) >= self.rpm_limit or
               self.current_token_sum + tokens >= self.tpm_limit):
            await asyncio.sleep(1.0)  # Longer sleep to reduce CPU usage
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            self.requests = [t for t in self.requests if t > minute_ago]
            self.tokens = [(t, tok)
                           for t, tok in self.tokens if t > minute_ago]
            self.current_token_sum = sum(tok for _, tok in self.tokens)

        # Record this request before acquiring semaphore
        self.requests.append(now)
        self.tokens.append((now, tokens))
        self.current_token_sum += tokens

        # Acquire semaphore for parallel request limiting
        try:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=30.0)
        except asyncio.TimeoutError:
            # Remove the recorded request if we timeout
            self.requests.pop()
            self.tokens.pop()
            self.current_token_sum -= tokens
            raise RuntimeError("Timeout waiting for rate limit semaphore")

    def release(self):
        """Release the semaphore for the next request"""
        try:
            self.semaphore.release()
        except ValueError:
            # Handle case where release is called without acquire
            pass


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


# Remove duplicated code block

    async def acquire(self, tokens: int):
        async with self.lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean up old entries and recalculate token sum
            self.requests = [t for t in self.requests if t > minute_ago]
            self.tokens = [(t, tok)
                           for t, tok in self.tokens if t > minute_ago]
            self.current_token_sum = sum(tok for _, tok in self.tokens)

            # Predictive check - will this request put us over the limit?
            while (len(self.requests) >= self.rpm_limit or
                   self.current_token_sum + tokens >= self.tpm_limit):
                await asyncio.sleep(1.0)  # Longer sleep to reduce CPU usage
                now = datetime.now()
                minute_ago = now - timedelta(minutes=1)
                self.requests = [t for t in self.requests if t > minute_ago]
                self.tokens = [(t, tok)
                               for t, tok in self.tokens if t > minute_ago]
                self.current_token_sum = sum(tok for _, tok in self.tokens)

        # Record this request before acquiring semaphore
        self.requests.append(now)
        self.tokens.append((now, tokens))
        self.current_token_sum += tokens

        # Acquire semaphore for parallel request limiting
        try:
            await asyncio.wait_for(self.semaphore.acquire(), timeout=30.0)
        except asyncio.TimeoutError:
            # Remove the recorded request if we timeout
            self.requests.pop()
            self.tokens.pop()
            self.current_token_sum -= tokens
            raise RuntimeError("Timeout waiting for rate limit semaphore")

    def release(self):
        """Release the semaphore for the next request"""
        try:
            self.semaphore.release()
        except ValueError:
            # Handle case where release is called without acquire
            pass


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
