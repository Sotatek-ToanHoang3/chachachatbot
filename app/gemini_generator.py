# app/gemini_generator.py (or wherever your class lives)

import logging
import os
from collections.abc import Mapping
from typing import Any

from chatlib.chatbot import DialogueTurn
from chatlib.chatbot.generators import ResponseGenerator
from chatlib.utils.jinja_utils import convert_to_jinja_template
from app.guard.qwen3_guard import (
    _CONTROVERSIAL_LABEL,
    _UNSAFE_LABEL,
    get_qwen3_guard,
)

# Official Google Generative AI client
import google.generativeai as genai
from google.generativeai import types

logger = logging.getLogger(__name__)

_KEY_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _search_mapping_for_key(mapping: Mapping, path: tuple[str, ...] = ()) -> str | None:
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            found = _search_mapping_for_key(value, path + (str(key),))
            if found:
                return found
            continue

        if isinstance(value, str):
            key_upper = str(key).upper()
            composed_key = "_".join((*[segment.upper() for segment in path], key_upper)) if path else key_upper

            if key_upper in _KEY_ENV_NAMES or composed_key in _KEY_ENV_NAMES:
                return value

            if key_upper.endswith("API_KEY") or composed_key.endswith("API_KEY"):
                haystack = f"{composed_key}_{key_upper}"
                if any(token in haystack for token in ("GEMINI", "GOOGLE", "GENAI")):
                    return value
    return None


def _resolve_gemini_api_key() -> str | None:
    for key_name in _KEY_ENV_NAMES:
        value = os.environ.get(key_name)
        if value:
            return value

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    for key_name in _KEY_ENV_NAMES:
        value = os.environ.get(key_name)
        if value:
            return value

    try:
        import streamlit as st  # type: ignore

        secrets = getattr(st, "secrets", None)
        if isinstance(secrets, Mapping):
            found = _search_mapping_for_key(secrets)
            if found:
                for key_name in _KEY_ENV_NAMES:
                    os.environ.setdefault(key_name, found)
                return found
    except Exception:
        pass

    return None


def _handle_missing_key(message: str) -> None:
    try:
        import streamlit as st  # type: ignore

        st.error(message)
        st.stop()
    except Exception:
        raise RuntimeError(message)


class GeminiGenerator(ResponseGenerator):
    def __init__(self, base_instruction: str, special_tokens=None, model_name: str | None = None):
        super().__init__()
        # Use the official client instead of the chatlib GeminiAPI wrapper
        api_key = _resolve_gemini_api_key()
        if not api_key:
            _handle_missing_key(
                "Gemini API key not found. Add GEMINI_API_KEY (or GOOGLE_API_KEY) either as an environment "
                "variable, in a local .env file, or via Streamlit secrets."
            )
            return  # st.stop() above prevents execution; the return keeps type-checkers happy.
        # Ensure dependent integrations (chatlib) see the same key.
        os.environ.setdefault("GEMINI_API_KEY", api_key)
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        os.environ.setdefault("GEMINI_COMPLETION_MODEL", model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash")
        genai.configure(api_key=api_key)

        # Model: pick a solid default; allow override via ctor or env.
        self.model_name = model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        self._guard = get_qwen3_guard()

        # Store base instruction and compile to a Jinja template
        self._template_str = base_instruction
        self._template = convert_to_jinja_template(base_instruction)

        self.special_tokens = special_tokens or {}
        self._params = {}

    def update_instruction_parameters(self, params: dict):
        self._params.update(params)

    async def _get_response_impl(self, dialogue: list[DialogueTurn], dry: bool = False):
        # Build context for your base instruction template
        context = dict(self._params)
        guard_metadata: dict[str, Any] = {}
        prompt_risk_label: str | None = None
        last_user_turn: DialogueTurn | None = None
        if dialogue:
            last_turn = dialogue[-1]
            last_user_turn = next((turn for turn in reversed(dialogue) if turn.is_user), None)
            context.update(
                {
                    "user_name": getattr(last_turn, "user_name", None),
                    "user_age": getattr(last_turn, "user_age", None),
                    "locale": getattr(last_turn, "locale", None),
                }
            )

        if last_user_turn is not None:
            should_moderate_prompt = True
            locale = context.get("locale") or getattr(last_user_turn, "locale", None)
            try:
                metadata = getattr(last_user_turn, "metadata", None) or {}
                if metadata.get("selected_emotions"):
                    should_moderate_prompt = False
                elif metadata.get("hide") is True and not last_user_turn.message.strip():
                    should_moderate_prompt = False
            except AttributeError:
                should_moderate_prompt = True

            if should_moderate_prompt:
                try:
                    prompt_decision = await self._guard.moderate_prompt(last_user_turn.message)
                    guard_metadata["prompt"] = prompt_decision.to_dict()
                    prompt_risk_label = prompt_decision.label
                except Exception:
                    logger.exception("Qwen3Guard prompt moderation failed.")
                else:
                    if prompt_risk_label == _UNSAFE_LABEL:
                        warning = self._build_safety_support_message(prompt_decision, locale)
                        metadata = {"moderation": guard_metadata, "safety_intervention": {
                            "type": "prompt",
                            "categories": prompt_decision.categories,
                            "label": prompt_decision.label,
                        }, "flagged_emotion": "negative"} if guard_metadata else {"flagged_emotion": "negative"}
                        return warning, metadata

        # Render the system instruction
        system_instruction = self._template.render(**context)
        if prompt_risk_label == _CONTROVERSIAL_LABEL:
            system_instruction = (
                f"{system_instruction}\n\n"
                "Safety Priority: The child may describe bullying or anger. Validate their feelings, "
                "discourage retaliation, and encourage involving a trusted adult for support."
            )

        # Convert dialogue to Google "contents" and prepend dynamic system instruction.
        contents: list[dict] = []
        for turn in dialogue:
            role = "user" if turn.is_user else "model"
            contents.append({"role": role, "parts": [{"text": turn.message}]})

        # Seed the conversation when initializing from an empty dialog.
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Start the session."}]}]

        # Gemini v1beta accepts only user/model roles. Prepend the instruction as a synthetic user turn.
        contents.insert(0, {"role": "user", "parts": [{"text": f"[Instruction]\\n{system_instruction}"}]})

        if dry:
            metadata = {"moderation": guard_metadata} if guard_metadata else {}
            return "...", metadata

        # Generation config; put the system prompt in the config (recommended style)
        generation_config = types.GenerationConfig(
            temperature=0.7,
            top_p=1.0,
            top_k=1,
            max_output_tokens=2048,
        )

        # Call the API (use async method to avoid blocking the event loop)
        model = genai.GenerativeModel(self.model_name)
        resp = await model.generate_content_async(
            contents=contents,
            generation_config=generation_config,
        )

        # The new client exposes resp.text; fallback to candidates if needed
        response_text = ""
        try:
            response_text = (resp.text or "").strip()
        except ValueError:
            response_text = ""

        if not response_text and getattr(resp, "candidates", None):
            for candidate in resp.candidates:
                parts = getattr(candidate, "content", None)
                if parts and getattr(parts, "parts", None):
                    text_parts = [getattr(part, "text", "") for part in parts.parts if getattr(part, "text", "")]
                    if text_parts:
                        response_text = " ".join(text_parts).strip()
                        break

        if not response_text and getattr(resp, "candidates", None):
            try:
                response_text = resp.candidates[0].content.parts[0].text.strip()
            except Exception:
                response_text = ""

        if last_user_turn is not None and response_text:
            try:
                stream_decision = await self._guard.moderate_response_stream(last_user_turn.message, response_text)
                guard_metadata["response"] = stream_decision.to_dict()
            except Exception:
                logger.exception("Qwen3Guard stream moderation failed.")
            else:
                if stream_decision.risk_level == _UNSAFE_LABEL:
                    warning = self._build_safety_support_message(stream_decision, context.get("locale"))
                    metadata = {"moderation": guard_metadata, "safety_intervention": {
                        "type": "response",
                        "category": stream_decision.category,
                        "risk_level": stream_decision.risk_level,
                    }, "flagged_emotion": "negative"} if guard_metadata else {"flagged_emotion": "negative"}
                    return warning, metadata

        metadata = {"moderation": guard_metadata} if guard_metadata else {}
        return response_text, metadata

    def restore_from_json(self, json_str: str):
        pass  # No state to restore

    def write_to_json(self) -> str:
        return "{}"  # No state to save

    @staticmethod
    def _build_safety_support_message(decision: Any, locale: str | None) -> str:
        categories: list[str] = []
        if hasattr(decision, "categories"):
            categories = [c for c in getattr(decision, "categories") or [] if c]
        elif hasattr(decision, "category") and getattr(decision, "category"):
            categories = [getattr(decision, "category")]

        if locale == "kr":
            return GeminiGenerator._build_korean_safety_message(categories)
        return GeminiGenerator._build_english_safety_message(categories)

    @staticmethod
    def _build_english_safety_message(categories: list[str]) -> str:
        # Craft an empathetic yet firm response without apologising.
        topics = ", ".join(categories) if categories else "something unsafe"
        return (
            f"I can tell you’re feeling really intense about {topics}. I want to keep everyone safe, "
            "so let’s slow down and talk about what’s making you feel this way."
        )

    @staticmethod
    def _build_korean_safety_message(categories: list[str]) -> str:
        topics = ", ".join(categories) if categories else "위험한 생각"
        return (
            f"{topics}에 대해 많이 화가 나고 힘들게 느끼는 것 같아. 모두가 안전할 수 있도록 잠시 진정하고 "
            "지금 어떤 일이 있었는지 이야기해줄래?"
        )
