import json
import logging
import sys
from types import ModuleType, SimpleNamespace

from fin_rag.config import Settings
from fin_rag.llm import _normalize_api_base, build_llm


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_build_llm_completes_against_openai_compatible_payload(monkeypatch, caplog):
    fake_modules = {}

    root_module = ModuleType("llama_index")
    root_module.__path__ = []
    core_module = ModuleType("llama_index.core")
    core_module.__path__ = []
    base_module = ModuleType("llama_index.core.base")
    base_module.__path__ = []
    llms_module = ModuleType("llama_index.core.llms")
    llms_module.__path__ = []
    monkeypatch.setitem(sys.modules, "llama_index", root_module)
    monkeypatch.setitem(sys.modules, "llama_index.core", core_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.base", base_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.llms", llms_module)

    types_module = ModuleType("llama_index.core.base.llms.types")

    class LLMMetadata:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class CompletionResponse:
        def __init__(self, text, delta=None, raw=None):
            self.text = text
            self.delta = delta
            self.raw = raw

    types_module.LLMMetadata = LLMMetadata
    types_module.CompletionResponse = CompletionResponse
    fake_modules["llama_index.core.base.llms.types"] = types_module

    callbacks_module = ModuleType("llama_index.core.callbacks")
    callbacks_module.CallbackManager = lambda *args, **kwargs: SimpleNamespace()
    fake_modules["llama_index.core.callbacks"] = callbacks_module

    completion_callbacks_module = ModuleType("llama_index.core.llms.callbacks")
    completion_callbacks_module.llm_completion_callback = lambda: (lambda func: func)
    fake_modules["llama_index.core.llms.callbacks"] = completion_callbacks_module

    custom_module = ModuleType("llama_index.core.llms.custom")

    class CustomLLM:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    custom_module.CustomLLM = CustomLLM
    fake_modules["llama_index.core.llms.custom"] = custom_module

    llm_module = ModuleType("llama_index.core.llms.llm")
    llm_module.CompletionToPromptType = object
    llm_module.MessagesToPromptType = object
    fake_modules["llama_index.core.llms.llm"] = llm_module

    types_module2 = ModuleType("llama_index.core.types")
    class PydanticProgramMode:
        DEFAULT = "default"
    types_module2.PydanticProgramMode = PydanticProgramMode
    fake_modules["llama_index.core.types"] = types_module2

    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr("fin_rag.llm.urlopen", lambda request, timeout: _FakeHTTPResponse({"choices": [{"message": {"content": "可以"}}]}))

    settings = Settings(
        llm_base_url="http://example.test/v1",
        llm_model="deepseek-chat",
        llm_api_key="sk-test-123456",
    )
    llm = build_llm(settings)

    with caplog.at_level(logging.INFO, logger="fin_rag.llm"):
        response = llm.complete("hello")

    assert response.text == "可以"
    assert response.raw["choices"][0]["message"]["content"] == "可以"
    assert llm.metadata.is_chat_model is True
    assert '"event": "llm_request_start"' in caplog.text
    assert '"event": "llm_request_success"' in caplog.text
    assert "hello" not in caplog.text
    assert "sk-test-123456" not in caplog.text


def test_normalize_api_base_appends_v1_once():
    assert _normalize_api_base("https://api.deepseek.com") == "https://api.deepseek.com/v1"
    assert _normalize_api_base("https://api.deepseek.com/v1") == "https://api.deepseek.com/v1"
