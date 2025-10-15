from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "Qwen3Guard moderation requires PyTorch. Install it via `pip install torch` before using the guard."
    ) from exc
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
try:
    import accelerate  # type: ignore  # noqa: F401
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False

logger = logging.getLogger(__name__)

_SAFE_LABEL = "Safe"
_CONTROVERSIAL_LABEL = "Controversial"
_UNSAFE_LABEL = "Unsafe"


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


@dataclass(slots=True)
class GuardDecision:
    label: Optional[str]
    categories: List[str] = field(default_factory=list)
    refusal: Optional[str] = None
    raw_output: str = ""

    @property
    def allowed(self) -> bool:
        return self.label == _SAFE_LABEL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "categories": list(self.categories),
            "refusal": self.refusal,
            "raw_output": self.raw_output,
        }


@dataclass(slots=True)
class StreamGuardDecision:
    risk_level: Optional[str]
    category: Optional[str]
    raw_result: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.risk_level == _SAFE_LABEL

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "risk_level": self.risk_level,
            "category": self.category,
        }
        if self.raw_result:
            payload["raw_result"] = dict(self.raw_result)
        return payload


class Qwen3GuardModerator:
    """
    Wraps the Qwen3Guard generative and streaming moderation models for local inference.
    """

    def __init__(
        self,
        gen_model_name: Optional[str] = None,
        stream_model_name: Optional[str] = None,
        prompt_warning: Optional[str] = None,
        response_warning: Optional[str] = None,
        max_new_tokens: int = 128,
    ):
        self.gen_model_name = gen_model_name or os.getenv("QWEN3_GUARD_GEN_MODEL", "Qwen/Qwen3Guard-Gen-0.6B")
        self.stream_model_name = stream_model_name or os.getenv(
            "QWEN3_GUARD_STREAM_MODEL", "Qwen/Qwen3Guard-Stream-0.6B"
        )
        self.prompt_warning_template = (
            prompt_warning
            or os.getenv(
                "QWEN3_GUARD_PROMPT_WARNING",
                "I’m sorry, but I can’t help with that request. Could you please ask in a different way?",
            )
        )
        self.response_warning_template = (
            response_warning
            or os.getenv(
                "QWEN3_GUARD_RESPONSE_WARNING",
                "I’m going to skip that answer so we can keep things safe. Let’s talk about something else.",
            )
        )
        self.max_new_tokens = max_new_tokens

        self._gen_model: Optional[AutoModelForCausalLM] = None
        self._gen_tokenizer: Optional[AutoTokenizer] = None
        self._stream_model: Optional[AutoModel] = None
        self._stream_tokenizer: Optional[AutoTokenizer] = None
        self._gen_unavailable = False
        self._stream_unavailable = False
        self._enabled = True
        self._prefer_cpu = False

        allow_cpu = os.getenv("QWEN3_GUARD_ALLOW_CPU", "0") == "1"
        if not torch.cuda.is_available() and not allow_cpu:
            logger.warning(
                "Qwen3Guard disabled: no CUDA device detected and QWEN3_GUARD_ALLOW_CPU not set. "
                "Set QWEN3_GUARD_ALLOW_CPU=1 to run moderation on CPU (can be slow)."
            )
            self._gen_unavailable = True
            self._stream_unavailable = True
            self._enabled = False

        self._gen_lock = threading.Lock()
        self._stream_lock = threading.Lock()

    # -------------------------- public API -------------------------- #

    async def moderate_prompt(self, prompt: str) -> GuardDecision:
        prompt = prompt.strip()
        if not prompt:
            return GuardDecision(label=_SAFE_LABEL)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._moderate_prompt_sync, prompt)

    async def moderate_response_stream(self, prompt: str, response: str) -> StreamGuardDecision:
        prompt = prompt.strip()
        response = response.strip()
        if not response:
            return StreamGuardDecision(risk_level=_SAFE_LABEL, category=None)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._stream_moderate_sync, prompt, response)

    def format_prompt_warning(self, decision: GuardDecision) -> str:
        return self._format_warning(self.prompt_warning_template, decision.categories)

    def format_response_warning(self, decision: StreamGuardDecision) -> str:
        categories = [decision.category] if decision.category else []
        return self._format_warning(self.response_warning_template, categories)

    # ----------------------- internal helpers ----------------------- #

    @staticmethod
    def _get_model_device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device

    def _ensure_gen_model(self) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        if self._gen_model is not None and self._gen_tokenizer is not None:
            return self._gen_tokenizer, self._gen_model
        with self._gen_lock:
            if self._gen_model is None or self._gen_tokenizer is None:
                logger.info("Loading Qwen3Guard generative model: %s", self.gen_model_name)
                tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name, trust_remote_code=False)
                # Prefer device_map auto only when accelerate is available; otherwise fall back to CPU to avoid dependency errors.
                device_map = "auto" if _HAS_ACCELERATE and not self._prefer_cpu else None
                use_cuda = torch.cuda.is_available() and not self._prefer_cpu
                dtype = torch.float16 if use_cuda else torch.float32
                model = AutoModelForCausalLM.from_pretrained(
                    self.gen_model_name,
                    dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=not _HAS_ACCELERATE,
                )
                if device_map is None:
                    target_device = torch.device("cuda" if use_cuda else "cpu")
                    model = model.to(target_device)
                self._gen_tokenizer = tokenizer
                self._gen_model = model
        return self._gen_tokenizer, self._gen_model  # type: ignore[return-value]

    def _ensure_stream_model(self) -> tuple[AutoTokenizer, AutoModel]:
        if self._stream_model is not None and self._stream_tokenizer is not None:
            return self._stream_tokenizer, self._stream_model
        with self._stream_lock:
            if self._stream_model is None or self._stream_tokenizer is None:
                logger.info("Loading Qwen3Guard stream model: %s", self.stream_model_name)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.stream_model_name, trust_remote_code=True
                )
                use_cuda = torch.cuda.is_available() and not self._prefer_cpu
                stream_dtype = torch.bfloat16 if use_cuda else torch.float32
                device_map = "auto" if _HAS_ACCELERATE and not self._prefer_cpu else None
                model = AutoModel.from_pretrained(
                    self.stream_model_name,
                    device_map=device_map,
                    dtype=stream_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=not _HAS_ACCELERATE,
                )
                if device_map is None:
                    target_device = torch.device("cuda" if use_cuda else "cpu")
                    model = model.to(target_device)
                model = model.eval()
                self._stream_tokenizer = tokenizer
                self._stream_model = model
        return self._stream_tokenizer, self._stream_model  # type: ignore[return-value]

    def _moderate_prompt_sync(self, prompt: str) -> GuardDecision:
        if self._gen_unavailable:
            return GuardDecision(label=_SAFE_LABEL)
        try:
            tokenizer, model = self._ensure_gen_model()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Qwen3Guard prompt moderation disabled after load failure.", exc_info=exc)
            self._gen_unavailable = True
            return GuardDecision(label=_SAFE_LABEL)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([text], return_tensors="pt")
        model_inputs = model_inputs.to(self._get_model_device(model))

        try:
            generated = model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        except RuntimeError as exc:
            if not self._prefer_cpu and ("cudnn" in str(exc).lower() or "nvrtc" in str(exc).lower()):
                logger.warning("Qwen3Guard prompt moderation failed on GPU, retrying on CPU.", exc_info=exc)
                self._prefer_cpu = True
                self._gen_model = None
                self._stream_model = None
                return self._moderate_prompt_sync(prompt)
            logger.warning("Qwen3Guard prompt moderation disabled after runtime failure.", exc_info=exc)
            self._gen_unavailable = True
            return GuardDecision(label=_SAFE_LABEL)
        output_ids = generated[0][model_inputs["input_ids"].shape[1]:]
        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        label, categories = self._extract_label_and_categories(content)
        categories = _dedupe(categories)

        return GuardDecision(
            label=label,
            categories=categories,
            raw_output=content.strip(),
        )

    def _stream_moderate_sync(self, prompt: str, response: str) -> StreamGuardDecision:
        if self._stream_unavailable:
            return StreamGuardDecision(risk_level=_SAFE_LABEL, category=None)
        try:
            tokenizer, model = self._ensure_stream_model()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Qwen3Guard stream moderation disabled after load failure.", exc_info=exc)
            self._stream_unavailable = True
            return StreamGuardDecision(risk_level=_SAFE_LABEL, category=None)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = tokenizer(text, return_tensors="pt")
        token_ids = inputs["input_ids"][0].to(self._get_model_device(model))

        token_ids_list = token_ids.tolist()
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        user_id = tokenizer.convert_tokens_to_ids("user")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        last_start = next(
            (i for i in range(len(token_ids_list) - 1, -1, -1) if token_ids_list[i : i + 2] == [im_start_id, user_id]),
            None,
        )
        if last_start is None:
            logger.warning("Unable to locate user turn boundary in token stream; defaulting to Safe.")
            return StreamGuardDecision(risk_level=_SAFE_LABEL, category=None)

        user_end_index = next(
            (i for i in range(last_start + 2, len(token_ids_list)) if token_ids_list[i] == im_end_id),
            None,
        )
        if user_end_index is None:
            logger.warning("Unable to locate user turn terminator in token stream; defaulting to Safe.")
            return StreamGuardDecision(risk_level=_SAFE_LABEL, category=None)

        # Moderate the user turn in one pass to seed the stream state.
        stream_state = None
        try:
            result, stream_state = model.stream_moderate_from_ids(
                token_ids[: user_end_index + 1],
                role="user",
                stream_state=None,
            )
        except RuntimeError as exc:
            if not self._prefer_cpu and ("cudnn" in str(exc).lower() or "nvrtc" in str(exc).lower()):
                logger.warning("Qwen3Guard stream moderation failed on GPU, retrying on CPU.", exc_info=exc)
                self._prefer_cpu = True
                self._stream_model = None
                self._gen_model = None
                return self._stream_moderate_sync(prompt, response)
            logger.warning("Qwen3Guard stream moderation disabled after runtime failure.", exc_info=exc)
            self._stream_unavailable = True
            return StreamGuardDecision(risk_level=_SAFE_LABEL, category=None)
        last_result = result
        for idx in range(user_end_index + 1, len(token_ids_list)):
            current_token = token_ids[idx]
            try:
                result, stream_state = model.stream_moderate_from_ids(
                    current_token,
                    role="assistant",
                    stream_state=stream_state,
                )
            except RuntimeError as exc:
                if not self._prefer_cpu and ("cudnn" in str(exc).lower() or "nvrtc" in str(exc).lower()):
                    logger.warning("Qwen3Guard stream moderation failed mid-stream on GPU, retrying on CPU.", exc_info=exc)
                    self._prefer_cpu = True
                    self._stream_model = None
                    self._gen_model = None
                    return self._stream_moderate_sync(prompt, response)
                logger.warning("Qwen3Guard stream moderation disabled after runtime failure.", exc_info=exc)
                self._stream_unavailable = True
                return StreamGuardDecision(risk_level=_SAFE_LABEL, category=None)
            last_result = result

        if hasattr(model, "close_stream"):
            try:
                model.close_stream(stream_state)
            except Exception:
                logger.debug("Failed to close Qwen3Guard stream state cleanly.", exc_info=True)

        risk = None
        category = None
        if last_result:
            risk = last_result.get("risk_level", [None])[-1]
            category = last_result.get("category", [None])[-1]

        return StreamGuardDecision(
            risk_level=risk,
            category=category,
            raw_result=last_result or {},
        )

    @staticmethod
    def _extract_label_and_categories(content: str) -> tuple[Optional[str], List[str]]:
        label_match = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", content)
        labels = re.findall(
            r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)",
            content,
        )
        label = label_match.group(1) if label_match else None
        categories = [c for c in labels if c != "None"]
        return label, categories

    def _format_warning(self, template: str, categories: List[str]) -> str:
        if not categories:
            return template
        category_text = ", ".join(categories)
        return f"{template}\n\n(Flagged topics: {category_text}.)"


_singleton_guard: Optional[Qwen3GuardModerator] = None
_singleton_lock = threading.Lock()


def get_qwen3_guard() -> Qwen3GuardModerator:
    global _singleton_guard
    if _singleton_guard is not None:
        return _singleton_guard
    with _singleton_lock:
        if _singleton_guard is None:
            _singleton_guard = Qwen3GuardModerator()
    return _singleton_guard
