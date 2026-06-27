from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fin_rag.config import Settings

logger = logging.getLogger(__name__)


def build_llm(settings: Settings):
    try:
        from llama_index.core.base.llms.types import LLMMetadata
        from llama_index.core.callbacks import CallbackManager
        from llama_index.core.llms.callbacks import llm_completion_callback
        from llama_index.core.llms.custom import CustomLLM
        from llama_index.core.llms.llm import CompletionToPromptType, MessagesToPromptType
        from llama_index.core.types import PydanticProgramMode
    except ModuleNotFoundError as exc:
        raise RuntimeError("llama-index llm dependencies are not installed.") from exc

    if not settings.llm_base_url.strip():
        raise RuntimeError("llm_base_url is not configured.")

    class OpenAICompatibleLLM(CustomLLM):
        model_name: str
        api_base: str
        api_key: str | None = None
        timeout: float = 30.0
        max_tokens: int = 1024
        temperature: float = 0.1

        def __init__(
            self,
            model_name: str,
            api_base: str,
            api_key: str | None = None,
            timeout: float = 30.0,
            max_tokens: int = 1024,
            temperature: float = 0.1,
            callback_manager: CallbackManager | None = None,
            system_prompt: str | None = None,
            messages_to_prompt: MessagesToPromptType | None = None,
            completion_to_prompt: CompletionToPromptType | None = None,
            pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        ) -> None:
            super().__init__(
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                callback_manager=callback_manager or CallbackManager([]),
                system_prompt=system_prompt,
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt,
                pydantic_program_mode=pydantic_program_mode,
            )

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=8192,
                num_output=self.max_tokens,
                is_chat_model=True,
                model_name=self.model_name,
            )

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            payload = self._post_chat_completion(prompt)
            text = payload["choices"][0]["message"]["content"]
            return self._build_completion_response(text=text, raw=payload)

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            response = self.complete(prompt, formatted=formatted, **kwargs)
            yield self._build_completion_response(text=response.text or "", delta=response.text or "", raw=response.raw)

        def _post_chat_completion(self, prompt: str) -> dict[str, object]:
            endpoint = f"{_normalize_api_base(self.api_base)}/chat/completions"
            started_at = time.perf_counter()
            logger.info(
                json.dumps(
                    {
                        "event": "llm_request_start",
                        "model": self.model_name,
                        "endpoint": endpoint,
                        "prompt_chars": len(prompt),
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                    },
                    ensure_ascii=False,
                )
            )
            body = json.dumps(
                {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False,
                },
                ensure_ascii=False,
            ).encode("utf-8")
            request = Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    **(
                        {"Authorization": f"Bearer {self.api_key}"}
                        if self.api_key
                        else {}
                    ),
                },
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                logger.info(
                    json.dumps(
                        {
                            "event": "llm_request_success",
                            "model": self.model_name,
                            "endpoint": endpoint,
                            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                            "response_chars": len(json.dumps(payload, ensure_ascii=False)),
                        },
                        ensure_ascii=False,
                    )
                )
                return payload
            except HTTPError as exc:
                logger.error(
                    json.dumps(
                        {
                            "event": "llm_request_error",
                            "model": self.model_name,
                            "endpoint": endpoint,
                            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                            "error_type": type(exc).__name__,
                            "http_status": exc.code,
                        },
                        ensure_ascii=False,
                    )
                )
                raise RuntimeError(f"LLM request failed with HTTP {exc.code}") from exc
            except URLError as exc:
                logger.error(
                    json.dumps(
                        {
                            "event": "llm_request_error",
                            "model": self.model_name,
                            "endpoint": endpoint,
                            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                            "error_type": type(exc).__name__,
                            "reason": str(exc.reason),
                        },
                        ensure_ascii=False,
                    )
                )
                raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        def _build_completion_response(self, text: str, raw: object, delta: str | None = None):
            from llama_index.core.base.llms.types import CompletionResponse

            return CompletionResponse(text=text, delta=delta, raw=raw)

    return OpenAICompatibleLLM(
        model_name=settings.llm_model,
        api_base=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )


def _normalize_api_base(api_base: str) -> str:
    from urllib.parse import urlparse, urlunparse

    normalized = api_base.rstrip("/")
    parsed = urlparse(normalized)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        return normalized
    new_path = f"{path}/v1" if path else "/v1"
    return urlunparse(parsed._replace(path=new_path))
