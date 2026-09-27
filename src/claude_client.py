"""
claude_client.py
=================

Thin wrapper around the Anthropic Messages API. The API key is always
read from the ``ANTHROPIC_API_KEY`` environment variable — never
hardcoded or logged.
"""

from __future__ import annotations

import logging
import os

import anthropic

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 4096


class ClaudeClientError(RuntimeError):
    """Raised when the Claude API cannot be reached or returns an error."""


class ClaudeClient:
    """A minimal, testable wrapper around :class:`anthropic.Anthropic`.

    Args:
        model: The Claude model name to call.
        max_tokens: The maximum number of tokens to generate per reply.
        api_key: Optional explicit API key. If omitted, read from the
            ``ANTHROPIC_API_KEY`` environment variable.

    Raises:
        ClaudeClientError: If no API key is available from either source.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ClaudeClientError(
                "No Anthropic API key found. Set the ANTHROPIC_API_KEY "
                "environment variable before starting the agent."
            )
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=resolved_key)

    def complete(self, system_prompt: str, user_message: str) -> str:
        """Get a single assistant reply for one system + user message pair.

        Args:
            system_prompt: Instructions describing the task.
            user_message: The user-provided content.

        Returns:
            The assistant's reply text.

        Raises:
            ClaudeClientError: If the API call fails for any reason.
        """
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.APIError as exc:
            logger.error("Claude API call failed: %s", exc)
            raise ClaudeClientError(f"Claude API call failed: {exc}") from exc

        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "".join(text_blocks)
