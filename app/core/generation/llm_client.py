"""
app/core/generation/llm_client.py

Async Gemini LLM client wrapper.

Wraps google-generativeai with:
  - Singleton model instances (loaded once at startup)
  - Async generation with generate_content_async
  - Tenacity retry: 3 attempts, exponential backoff (2s → 4s → 8s)
  - Token usage logging on every call
  - Structured response: text + token_count

Two model roles (from settings):
  llm_model_router : fast calls — Prompt 1 (router)
  llm_model_main   : quality calls — Prompt 2 (HyDE) + Prompt 4 (answer)

Usage:
    from app.core.generation.llm_client import get_llm_client
    client = get_llm_client()

    # Single-turn call (router, HyDE)
    response = await client.complete(
        prompt="...",
        role="router",          # uses llm_model_router
    )

    # Answer generation (with system prompt injected automatically)
    response = await client.complete(
        prompt="...",
        role="main",            # uses llm_model_main
        use_system_prompt=True,
    )

    print(response.text)
    print(response.token_count)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from app.config.settings import get_settings
from app.core.generation.prompts.system_prompt import get_system_prompt
from app.utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

ModelRole = Literal["main", "router"]


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """Structured response from a Gemini LLM call."""
    text: str
    token_count: int          # total tokens used (prompt + output)
    prompt_tokens: int = 0    # input tokens
    output_tokens: int = 0    # generated tokens
    model: str = ""
    role: ModelRole = "main"

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT
# ══════════════════════════════════════════════════════════════════════════════

class GeminiClient:
    """
    Async wrapper around google-generativeai.

    Maintains two GenerativeModel instances:
      _router_model : for Prompt 1 (router) — fast
      _main_model   : for Prompt 2 (HyDE) + Prompt 4 (answer) — quality

    Both are initialised lazily on first use (not at import time).
    System prompt is injected into _main_model at construction time.
    """

    def __init__(self) -> None:
        self._router_model = None
        self._main_model = None
        self._initialised = False

    def _initialise(self) -> None:
        """Load Gemini models. Called once on first complete() call."""
        if self._initialised:
            return

        try:
            import google.generativeai as genai
        except ImportError:
            log.error(
                "google_generativeai_not_installed",
                fix="pip install google-generativeai",
            )
            raise

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set in .env. "
                "Get your key from https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=settings.gemini_api_key)

        # Router model — no system prompt (prompt is self-contained)
        self._router_model = genai.GenerativeModel(
            model_name=settings.llm_model_router,
        )

        # Main model — system prompt injected at construction
        self._main_model = genai.GenerativeModel(
            model_name=settings.llm_model_main,
            system_instruction=get_system_prompt(),
        )

        self._initialised = True
        log.info(
            "gemini_models_loaded",
            router_model=settings.llm_model_router,
            main_model=settings.llm_model_main,
        )

    def _get_model(self, role: ModelRole):
        self._initialise()
        return self._router_model if role == "router" else self._main_model

    def _build_generation_config(self, role: ModelRole, max_tokens: Optional[int] = None):
        """Build Gemini GenerationConfig for a given role."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise

        return genai.types.GenerationConfig(
            max_output_tokens=max_tokens or settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            candidate_count=1,
        )

    # ── Core retry wrapper ────────────────────────────────────────────────────

    async def _call_with_retry(
        self,
        model,
        prompt: str,
        generation_config,
    ):
        """
        Call Gemini with tenacity retry.
        3 attempts, exponential backoff: 2s → 4s → 8s.
        Retries on any exception (rate limits, transient errors, timeouts).
        """

        @retry(
            stop=stop_after_attempt(settings.llm_max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _attempt():
            return await asyncio.wait_for(
                model.generate_content_async(
                    contents=prompt,
                    generation_config=generation_config,
                ),
                timeout=settings.llm_timeout_seconds,
            )

        return await _attempt()

    # ── Public API ────────────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        role: ModelRole = "main",
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send a prompt to Gemini and return a structured LLMResponse.

        Args:
            prompt:     Complete prompt string (system prompt already injected
                        into the model for role='main').
            role:       'main' → llm_model_main (answer generation, HyDE)
                        'router' → llm_model_router (query classification)
            max_tokens: Override max output tokens. None = use settings value.

        Returns:
            LLMResponse with .text and .token_count populated.

        Raises:
            Exception if all retry attempts fail.
        """
        model = self._get_model(role)
        gen_config = self._build_generation_config(role, max_tokens)

        try:
            response = await self._call_with_retry(model, prompt, gen_config)
        except Exception as exc:
            log.error(
                "llm_call_failed",
                role=role,
                model=settings.llm_model_main if role == "main" else settings.llm_model_router,
                error=str(exc),
                prompt_preview=prompt[:100],
            )
            raise

        # Extract text
        text = ""
        try:
            text = response.text
        except (AttributeError, ValueError):
            # Handle blocked/empty responses
            if response.candidates:
                parts = response.candidates[0].content.parts
                text = " ".join(p.text for p in parts if hasattr(p, "text"))
            log.warning("llm_response_extraction_fallback", role=role)

        # Extract token usage
        prompt_tokens = 0
        output_tokens = 0
        total_tokens  = 0
        try:
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens  = usage.total_token_count or (prompt_tokens + output_tokens)
        except AttributeError:
            total_tokens = len(prompt.split()) + len(text.split())  # rough fallback

        model_name = settings.llm_model_main if role == "main" else settings.llm_model_router

        log.info(
            "llm_call_complete",
            role=role,
            model=model_name,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            text=text.strip(),
            token_count=total_tokens,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            model=model_name,
            role=role,
        )

    async def count_tokens(self, text: str, role: ModelRole = "main") -> int:
        """
        Count tokens for a text string using Gemini's token counter.
        More accurate than tiktoken for Gemini models.
        Falls back to approximate count on error.
        """
        try:
            model = self._get_model(role)
            result = await asyncio.get_event_loop().run_in_executor(
                None, model.count_tokens, text
            )
            return result.total_tokens
        except Exception:
            from app.utils.text_utils import count_tokens_approx
            return count_tokens_approx(text)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_client_instance: Optional[GeminiClient] = None


def get_llm_client() -> GeminiClient:
    """
    Return the shared GeminiClient singleton.
    Models are initialised lazily on first complete() call.

    Usage in any pipeline component:
        from app.core.generation.llm_client import get_llm_client
        client = get_llm_client()
        response = await client.complete(prompt, role="router")
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance