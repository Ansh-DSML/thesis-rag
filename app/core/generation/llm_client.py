"""
app/core/generation/llm_client.py

Async Gemini LLM client using the new google-genai SDK.
(google.generativeai is deprecated — this uses google.genai instead)

Two model roles:
  router : fast classification calls  (Prompt 1)
  main   : quality generation calls   (Prompt 2 + Prompt 4)

Free tier note:
  gemini-2.5-flash  → available free (5 RPM, 250K TPM, 20 RPD)
  gemini-2.5-pro    → NOT available on free tier (0/0 limits)
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


@dataclass
class LLMResponse:
    text: str
    token_count: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    role: ModelRole = "main"

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


class GeminiClient:
    """
    Async wrapper around google-genai SDK.
    Singleton — get via get_llm_client().
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
            raise ValueError("GEMINI_API_KEY not set in .env")

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._initialised = True
        log.info(
            "gemini_client_initialised",
            router_model=settings.llm_model_router,
            main_model=settings.llm_model_main,
        )

    def _get_model_name(self, role: ModelRole) -> str:
        return settings.llm_model_router if role == "router" else settings.llm_model_main

    async def _call_with_retry(
    self, model_name, prompt, system_instruction, max_tokens, disable_thinking=False
):
        from google.genai import types

        # Disable thinking for router — it produces truncated JSON
        # Keep thinking for main — produces better answers
        thinking_config = None
        if disable_thinking:
            thinking_config = types.ThinkingConfig(thinking_budget=0)

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=settings.llm_temperature,
            system_instruction=system_instruction,
            thinking_config=thinking_config,
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
                    model=model_name,
                    contents=prompt,
                    config=config,
                ),
                timeout=settings.llm_timeout_seconds,
            )

        return await _attempt()

    async def complete(
        self,
        prompt: str,
        role: ModelRole = "main",
        max_tokens: Optional[int] = None,
        disable_thinking: bool = False,      # ← ADD THIS PARAMETER
    ) -> LLMResponse:
        self._initialise()
        model_name   = self._get_model_name(role)
        system_instr = get_system_prompt() if role == "main" else None
        tokens       = max_tokens or settings.llm_max_tokens

        try:
            response = await self._call_with_retry(
                model_name=model_name,
                prompt=prompt,
                system_instruction=system_instr,
                max_tokens=tokens,
                disable_thinking=disable_thinking,   # ← ADD THIS
            )
        except Exception as exc:
            log.error(
                "llm_call_failed",
                role=role,
                model=model_name,
                error=str(exc),
                prompt_preview=prompt[:100],
            )
            raise

        # Extract text
        text = ""
        try:
            text = response.text or ""
        except Exception:
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text

        # Extract token usage
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
        try:
            self._initialise()
            result = await self._client.aio.models.count_tokens(
                model=self._get_model_name(role),
                contents=text,
            )
            return result.total_tokens
        except Exception:
            from app.utils.text_utils import count_tokens_approx
            return count_tokens_approx(text)


_client_instance: Optional[GeminiClient] = None


def get_llm_client() -> GeminiClient:
    """Return the shared GeminiClient singleton."""
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance