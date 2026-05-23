"""
app/core/generation/llm_client.py

Dual-provider LLM client:
  Router role → Gemini 2.5 Flash  (thinking disabled, fast JSON output)
  Main role   → Groq llama-3.3-70b-versatile  (full reasoning, no thinking budget issue)

Public API is identical to the single-provider version — nothing else in the
codebase needs to change. query_router.py, hyde.py, answer_generator.py all
continue calling complete(prompt, role=...) exactly as before.

Why two providers:
  Gemini 2.5 Flash thinking tokens consume the output budget → router JSON
  gets truncated to 17 tokens. Disabling thinking fixes the router.
  Groq has no thinking phase → llama-3.3-70b uses its full context for
  reasoning and generation with no token budget conflict.
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
)

from app.config.settings import get_settings
from app.core.generation.prompts.system_prompt import get_system_prompt
from app.utils.logger import get_logger

log      = get_logger(__name__)
settings = get_settings()

ModelRole = Literal["main", "router"]


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    text: str
    token_count: int   = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    model: str         = ""
    role: ModelRole    = "main"

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# GROQ CLIENT  — main model (answer generation + HyDE)
# ══════════════════════════════════════════════════════════════════════════════

class GroqLLMClient:
    """
    Async Groq client for llama-3.3-70b-versatile.
    Used for: Prompt 2 (HyDE) and Prompt 4 (answer generation).
    No thinking token conflict — Groq uses full context for reasoning.
    """

    def __init__(self) -> None:
        self._client      = None
        self._initialised = False

    def _initialise(self) -> None:
        if self._initialised:
            return
        try:
            from groq import AsyncGroq
        except ImportError:
            log.error("groq_not_installed", fix="pip install groq")
            raise

        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env. "
                "Get your key from https://console.groq.com/keys"
            )

        self._client = AsyncGroq(api_key=settings.groq_api_key)
        self._initialised = True
        log.info("groq_client_initialised", model=settings.llm_model_main)

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        use_system_prompt: bool = True,
    ) -> LLMResponse:
        self._initialise()

        messages = []
        if use_system_prompt:
            messages.append({"role": "system", "content": get_system_prompt()})
        messages.append({"role": "user", "content": prompt})

        @retry(
            stop=stop_after_attempt(settings.llm_max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _attempt():
            return await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=settings.llm_model_main,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=settings.llm_temperature,
                ),
                timeout=settings.llm_timeout_seconds,
            )

        try:
            response = await _attempt()
        except Exception as exc:
            log.error(
                "groq_call_failed",
                model=settings.llm_model_main,
                error=str(exc),
                prompt_preview=prompt[:100],
            )
            raise

        text          = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens        if response.usage else 0
        output_tokens = response.usage.completion_tokens    if response.usage else 0
        total_tokens  = response.usage.total_tokens         if response.usage else 0

        log.info(
            "llm_call_complete",
            role="main",
            model=settings.llm_model_main,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            text=text.strip(),
            token_count=total_tokens,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            model=settings.llm_model_main,
            role="main",
        )


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI CLIENT  — router only (Prompt 1, thinking disabled)
# ══════════════════════════════════════════════════════════════════════════════

class GeminiRouterClient:
    """
    Gemini 2.5 Flash client used ONLY for the query router (Prompt 1).
    Thinking is disabled so the full output budget goes to JSON generation.
    With thinking enabled, Flash uses ~480 tokens for thinking and leaves
    only 17 tokens for actual output → JSON truncation.
    """

    def __init__(self) -> None:
        self._client      = None
        self._initialised = False

    def _initialise(self) -> None:
        if self._initialised:
            return
        try:
            from google import genai
        except ImportError:
            log.error("google_genai_not_installed", fix="pip install google-genai")
            raise

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._initialised = True
        log.info("gemini_router_client_initialised", model=settings.llm_model_router)

    async def complete(self, prompt: str, max_tokens: int) -> LLMResponse:
        self._initialise()

        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=settings.llm_temperature,
            # No system_instruction — router prompt is fully self-contained
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # disable thinking
        )

        @retry(
            stop=stop_after_attempt(settings.llm_max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        async def _attempt():
            return await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=settings.llm_model_router,
                    contents=prompt,
                    config=config,
                ),
                timeout=settings.llm_timeout_seconds,
            )

        try:
            response = await _attempt()
        except Exception as exc:
            log.error(
                "gemini_router_call_failed",
                model=settings.llm_model_router,
                error=str(exc),
                prompt_preview=prompt[:100],
            )
            raise

        text = ""
        try:
            text = response.text or ""
        except Exception:
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text

        prompt_tokens = output_tokens = total_tokens = 0
        try:
            usage         = response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            total_tokens  = usage.total_token_count or (prompt_tokens + output_tokens)
        except Exception:
            total_tokens = len(prompt.split()) + len(text.split())

        log.info(
            "llm_call_complete",
            role="router",
            model=settings.llm_model_router,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        return LLMResponse(
            text=text.strip(),
            token_count=total_tokens,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            model=settings.llm_model_router,
            role="router",
        )


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED CLIENT  — public interface (unchanged API for all callers)
# ══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    Routes calls to the correct provider based on role:
      role="router" → GeminiRouterClient (Flash, thinking disabled)
      role="main"   → GroqLLMClient     (llama-3.3-70b, full reasoning)

    All callers (query_router.py, hyde.py, answer_generator.py) use this
    class via get_llm_client(). No changes needed in those files.
    """

    def __init__(self) -> None:
        self._groq   = GroqLLMClient()
        self._gemini = GeminiRouterClient()

    async def complete(
        self,
        prompt: str,
        role: ModelRole = "main",
        max_tokens: Optional[int] = None,
        disable_thinking: bool = False,   # kept for API compatibility, ignored
    ) -> LLMResponse:
        """
        Send a prompt to the appropriate LLM provider.

        role="router" → Gemini Flash (fast JSON classification)
        role="main"   → Groq llama-3.3-70b (HyDE + answer generation)
        """
        if role == "router":
            tokens = max_tokens or 500
            return await self._gemini.complete(prompt=prompt, max_tokens=tokens)
        else:
            tokens = max_tokens or settings.llm_max_tokens
            return await self._groq.complete(
                prompt=prompt,
                max_tokens=tokens,
                use_system_prompt=True,
            )

    async def count_tokens(self, text: str, role: ModelRole = "main") -> int:
        """Approximate token count."""
        from app.utils.text_utils import count_tokens_approx
        return count_tokens_approx(text)


# ── Singleton ─────────────────────────────────────────────────────────────────

_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return the shared LLMClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance