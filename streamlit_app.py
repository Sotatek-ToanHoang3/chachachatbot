import asyncio
from os import environ, getcwd, path

import streamlit as st
from dotenv import load_dotenv

from app.common import ChatbotLocale
from app.response_generator import EmotionChatbotResponseGenerator
from chatlib.chatbot import DialogueTurn


# -------------------------- environment bootstrap -------------------------- #
load_dotenv(path.join(getcwd(), ".env"))
if not environ.get("GOOGLE_API_KEY") and environ.get("GEMINI_API_KEY"):
    environ["GOOGLE_API_KEY"] = environ["GEMINI_API_KEY"]


# ------------------------------- UI helpers -------------------------------- #
def _create_generator(name: str, age: int, locale: ChatbotLocale) -> EmotionChatbotResponseGenerator:
    return EmotionChatbotResponseGenerator(user_name=name, user_age=age, locale=locale, verbose=False)


async def _generate_initial_turn(generator: EmotionChatbotResponseGenerator):
    response, metadata, _ = await generator.get_response([])
    return DialogueTurn(message=response, is_user=False, metadata=metadata)


async def _generate_reply(
    generator: EmotionChatbotResponseGenerator,
    dialogue: list[DialogueTurn],
) -> DialogueTurn:
    response, metadata, _ = await generator.get_response(dialogue)
    return DialogueTurn(message=response, is_user=False, metadata=metadata)


def _ensure_session_state() -> None:
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "dialogue" not in st.session_state:
        st.session_state.dialogue = []
    if "locale" not in st.session_state:
        st.session_state.locale = ChatbotLocale.English


def _reset_session():
    st.session_state.generator = None
    st.session_state.dialogue = []


def _rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ------------------------------- Streamlit UI ------------------------------ #
st.set_page_config(page_title="ChaCha Chatbot", page_icon="💬", layout="wide")
_ensure_session_state()

st.sidebar.header("Session setup")
with st.sidebar.form("session_setup"):
    user_name = st.text_input("Child's name", value=st.session_state.get("user_name", "Alex"))
    user_age = st.number_input("Child's age", min_value=0, max_value=120, value=int(st.session_state.get("user_age", 10)))
    locale_choice = st.selectbox("Language", options=["English", "Korean"], index=0 if st.session_state.locale == ChatbotLocale.English else 1)
    start_clicked = st.form_submit_button("Start / Restart")

if start_clicked:
    locale = ChatbotLocale.English if locale_choice == "English" else ChatbotLocale.Korean
    st.session_state.locale = locale
    st.session_state.user_name = user_name.strip()
    st.session_state.user_age = int(user_age)
    _reset_session()
    st.session_state.generator = _create_generator(
        name=st.session_state.user_name,
        age=st.session_state.user_age,
        locale=locale,
    )
    initial_turn = asyncio.run(_generate_initial_turn(st.session_state.generator))
    st.session_state.dialogue.append(initial_turn)
    _rerun_app()

if st.sidebar.button("Clear conversation", type="primary"):
    _reset_session()
    _rerun_app()

st.title("ChaCha – Emotional Support & Assignment Tutor")
st.caption("Share how you're feeling or ask for help with homework. ChaCha will respond in kid-friendly language and switch to tutor mode when appropriate.")

if st.session_state.generator is None:
    st.info("Configure the session in the sidebar and press **Start / Restart** to begin chatting.")
    st.stop()

for turn in st.session_state.dialogue:
    role = "user" if turn.is_user else "assistant"
    message_caption = None
    if not turn.is_user and turn.metadata is not None:
        moderation = turn.metadata.get("moderation")
        if moderation and "response" in moderation:
            decision = moderation["response"]
            if decision.get("risk_level") and decision["risk_level"] != "Safe":
                message_caption = f"⚠️ Moderation flagged: {decision['risk_level']}"
    with st.chat_message(role):
        st.markdown(turn.message)
        if message_caption:
            st.caption(message_caption)

user_prompt = st.chat_input("Type your message here...")
if user_prompt:
    user_prompt = user_prompt.strip()
    if user_prompt:
        st.session_state.dialogue.append(DialogueTurn(message=user_prompt, is_user=True))
        assistant_turn = asyncio.run(_generate_reply(st.session_state.generator, st.session_state.dialogue))
        st.session_state.dialogue.append(assistant_turn)
        _rerun_app()
