import re
from typing import Optional

from chatlib.utils import dict_utils
from chatlib.chatbot import ResponseGenerator, Dialogue, dialogue_utils
from chatlib.chatbot.generators import StateBasedResponseGenerator, StateType
from chatlib.chatbot.dialogue_to_csv import DialogueCSVWriter, TurnValueExtractor
from chatlib.chatbot.message_transformer import SpecialTokenExtractionTransformer

from app.common import EmotionChatbotPhase, SPECIAL_TOKEN_REGEX, SPECIAL_TOKEN_CONFIG, ChatbotLocale, FindDialogueSummarizerParams
import app.common
from app.gemini_generator import GeminiGenerator
from app.phases import explore, label, find, record, share, help, tutor


class EmotionChatbotResponseGenerator(StateBasedResponseGenerator[EmotionChatbotPhase]):

    def __init__(self,
                 user_name: str | None = None,
                 user_age: int = None,
                 locale: ChatbotLocale = ChatbotLocale.Korean,
                 verbose: bool = False):
        super().__init__(initial_state=EmotionChatbotPhase.Explore,
                         verbose=verbose,
                         message_transformers=[
                             SpecialTokenExtractionTransformer.remove_all_regex("clean_special_tokens",
                                                                                SPECIAL_TOKEN_REGEX)
                         ]
                         )

        self.__user_name = user_name
        self.__user_age = user_age
        self.__locale = locale
        self.__latest_feeling_sentiment: Optional[str] = None
        self.__latest_feeling_statement: Optional[str] = None
        self.__last_assignment_request: Optional[str] = None

        self.__generators: dict[EmotionChatbotPhase, GeminiGenerator] = dict()

        self.__generators[EmotionChatbotPhase.Explore] = explore.create_generator()
        self.__generators[EmotionChatbotPhase.Label] = label.create_generator()
        self.__generators[EmotionChatbotPhase.Find] = find.create_generator()
        self.__generators[EmotionChatbotPhase.Record] = record.create_generator()
        self.__generators[EmotionChatbotPhase.Share] = share.create_generator()
        self.__generators[EmotionChatbotPhase.Help] = help.create_generator()
        self.__generators[EmotionChatbotPhase.Tutor] = tutor.create_generator()

    def write_to_json(self, parcel: dict):
        super().write_to_json(parcel)
        parcel["user_name"] = self.__user_name
        parcel["user_age"] = self.__user_age
        parcel["locale"] = self.__locale

    def restore_from_json(self, parcel: dict):
        self.__user_name = parcel["user_name"]
        self.__user_age = parcel["user_age"]
        self.__locale = parcel["locale"] if "locale" in parcel else ChatbotLocale.Korean

        super().restore_from_json(parcel)

    @property
    def user_name(self)->str:
        return self.__user_name

    @property
    def user_age(self)->int:
        return self.__user_age

    @property
    def locale(self)->ChatbotLocale:
        return self.__locale

    @locale.setter
    def locale(self, locale: ChatbotLocale):
        self.__locale = locale
        self.__generators[self.current_state].update_instruction_parameters(dict(locale=locale))

    def get_generator(self, state: StateType, payload: dict | None) -> ResponseGenerator:
        # Get generator caches
        generator = self.__generators[state]

        if state == EmotionChatbotPhase.Explore:
            generator.update_instruction_parameters(dict(user_name=self.__user_name, user_age=self.__user_age,
                                                         locale=self.__locale,
                                                         revisited=True if payload is not None and payload[
                                                             "revisited"] is True else False))
        elif state == EmotionChatbotPhase.Label:
            generator.update_instruction_parameters(dict(**payload, locale=self.__locale))  # Put the result of rapport conversation
        elif state == EmotionChatbotPhase.Tutor:
            context = payload or {}
            generator.update_instruction_parameters(
                dict(
                    user_name=self.__user_name,
                    user_age=self.__user_age,
                    locale=self.__locale,
                    feeling_sentiment=context.get("feeling_sentiment"),
                    feeling_statement=context.get("feeling_statement"),
                    assignment_request=context.get("assignment_request"),
                )
            )
        elif state in [EmotionChatbotPhase.Find, EmotionChatbotPhase.Share, EmotionChatbotPhase.Record]:
            explore_payload = self._get_memoized_payload(EmotionChatbotPhase.Explore) or {}
            label_payload = self._get_memoized_payload(EmotionChatbotPhase.Label) or {}
            generator.update_instruction_parameters(
                dict(key_episode=explore_payload.get("key_episode"),
                     identified_emotions=label_payload.get("identified_emotions"),
                     locale=self.__locale
                     )
            )
        return generator

    def update_generator(self, generator: ResponseGenerator, payload: dict | None):
        if isinstance(generator, GeminiGenerator) and payload is not None:
            generator.update_instruction_parameters(dict(summarizer_result=payload))

    # ----------------------------- Moderation helpers ----------------------------- #

    def __track_feeling(self, message: str):
        sentiment = self.__classify_feeling(message)
        if sentiment:
            self.__latest_feeling_sentiment = sentiment
            self.__latest_feeling_statement = message.strip()

    @staticmethod
    def __clean_text(message: str) -> str:
        return re.sub(r"\s+", " ", message.lower()).strip()

    def __classify_feeling(self, message: str) -> Optional[str]:
        text = self.__clean_text(message)
        if not text:
            return None
        negative_patterns = [
            "not good",
            "not okay",
            "not ok",
            "not happy",
            "sad",
            "upset",
            "bad",
            "angry",
            "mad",
            "tired",
            "stressed",
            "worried",
            "anxious",
            "terrible",
            "awful",
            "depressed",
            "hurt someone",
            "hurt myself",
            "kill",
            "killing",
            "violent",
            "violence",
        ]
        if any(pattern in text for pattern in negative_patterns):
            return "negative"
        positive_patterns = [
            "happy",
            "great",
            "good",
            "awesome",
            "amazing",
            "excited",
            "fine",
            "okay",
            "ok",
            "cool",
            "pretty good",
            "doing well",
            "feeling good",
        ]
        for pattern in positive_patterns:
            if f"not {pattern}" in text or f"no {pattern}" in text:
                return "negative"
        if any(pattern in text for pattern in positive_patterns):
            return "positive"
        return None

    def __detect_assignment_request(self, message: str) -> bool:
        text = self.__clean_text(message)
        if not text:
            return False
        study_keywords = [
            "homework",
            "assignment",
            "essay",
            "project",
            "worksheet",
            "problem",
            "problem set",
            "question",
            "math",
            "science",
            "history",
            "report",
            "lab",
            "test",
            "exam",
            "quiz",
            "study guide",
        ]
        helper_keywords = ["help", "explain", "teach", "show", "understand", "solve", "work through", "walk me", "practice"]
        if any(kw in text for kw in study_keywords) and any(helper in text for helper in helper_keywords):
            return True
        if re.search(r"help\s+me\s+with\s+(my\s+)?(homework|assignment|math|science|essay|project)", text):
            return True
        if re.search(r"can\s+you\s+(help|explain|solve|check)\s", text) and any(kw in text for kw in study_keywords):
            return True
        return False

    def __should_enter_tutor(self, message: str) -> bool:
        if not self.__detect_assignment_request(message):
            return False
        return True

    def __should_exit_tutor(self, message: str) -> bool:
        text = self.__clean_text(message)
        if not text:
            return False
        if self.__detect_assignment_request(message):
            return False
        exit_markers = [
            "thanks",
            "thank you",
            "i'm done",
            "i am done",
            "that's all",
            "that is all",
            "no more questions",
            "no more homework",
            "i'm good now",
            "i am good now",
            "let's talk about feelings",
            "let us talk about feelings",
            "something else",
            "back to feelings",
            "finished",
            "got it now",
        ]
        return any(marker in text for marker in exit_markers)

    def __build_tutor_payload(self, current: EmotionChatbotPhase, request_message: str) -> dict:
        explore_payload = self._get_memoized_payload(EmotionChatbotPhase.Explore) or {}
        label_payload = self._get_memoized_payload(EmotionChatbotPhase.Label) or {}
        payload = {
            "key_episode": explore_payload.get("key_episode"),
            "user_emotion": self.__latest_feeling_statement or explore_payload.get("user_emotion"),
            "identified_emotions": label_payload.get("identified_emotions"),
            "feeling_sentiment": self.__latest_feeling_sentiment,
            "feeling_statement": self.__latest_feeling_statement,
            "assignment_request": request_message.strip(),
            "source_state": current.value if isinstance(current, EmotionChatbotPhase) else str(current),
        }
        self.__last_assignment_request = request_message.strip()
        return payload

    async def calc_next_state_info(self, current: EmotionChatbotPhase, dialog: Dialogue) -> tuple[
                                                                                                EmotionChatbotPhase | None, dict | None] | None:

        # dialog = dialogue_utils.extract_last_turn_sequence(dialog, lambda turn: dict_utils.get_nested_value(turn.metadata, "state") == current or turn.is_user)

        current_state_ai_turns = [turn for turn in StateBasedResponseGenerator.trim_dialogue_recent_n_states(dialog, 1)
                                  if
                                  turn.is_user == False]

        if len(dialog) == 0:
            return None
        last_user_turn, _ = dialogue_utils.find_last_turn(dialog, lambda turn: turn.is_user)
        last_user_message = last_user_turn.message if last_user_turn is not None else ""
        if last_user_turn is not None:
            self.__track_feeling(last_user_message)
        explore_payload = self._get_memoized_payload(EmotionChatbotPhase.Explore) or {}
        label_payload = self._get_memoized_payload(EmotionChatbotPhase.Label) or {}
        # Check if the user expressed sensitive topics
        summarizer_result = await help.summarizer.run(None, dialog, help.summarizer_params)
        if summarizer_result.sensitive_topic is True:
            return EmotionChatbotPhase.Help, None
        if current == EmotionChatbotPhase.Tutor:
            if last_user_turn is not None and self.__should_exit_tutor(last_user_message):
                return EmotionChatbotPhase.Explore, {"revisited": True}
            return None
        if last_user_turn is not None and self.__should_enter_tutor(last_user_message):
            tutor_payload = self.__build_tutor_payload(current, last_user_message)
            return EmotionChatbotPhase.Tutor, tutor_payload

        # Explore --> Label
        if current == EmotionChatbotPhase.Explore:
            # Minimum 3 rapport building conversation turns
            if len(current_state_ai_turns) >= 2:
                summarizer_result = await explore.summarizer.run(explore.summarizer_examples, dialog, explore.summarizer_params)
                print(summarizer_result)
                # print(f"Phase suggestion: {phase_suggestion}")
                if summarizer_result.move_to_next is True:
                    return EmotionChatbotPhase.Label, summarizer_result.model_dump()
                else:
                    return None, summarizer_result.model_dump()
        # Label --> Find OR Record
        elif current == EmotionChatbotPhase.Label:
            print("Current AI turns: ", len(current_state_ai_turns))
            summarizer_result = await label.summarizer.run(
                label.summarizer_examples,
                dialog,
                app.common.LabelDialogueSummarizerParams(
                    key_episode=explore_payload.get("key_episode"),
                    user_emotion=explore_payload.get("user_emotion"),
                ),
            )
            print(summarizer_result)

            if summarizer_result.next_phase == "find":
                if len(current_state_ai_turns) >= 3:
                    return EmotionChatbotPhase.Find, summarizer_result.model_dump()
            elif summarizer_result.next_phase == "record":
                if len(current_state_ai_turns) >= 3:
                    return EmotionChatbotPhase.Record, summarizer_result.model_dump()
            else:
                return None, summarizer_result.model_dump()
        # Find/Record --> Share
        elif current == EmotionChatbotPhase.Find or current == EmotionChatbotPhase.Record:

            summarizer = find.summarizer if current == EmotionChatbotPhase.Find else record.summarizer
            summarizer_result = await summarizer.run(
                find.summarizer_examples if current == EmotionChatbotPhase.Find else record.summarizer_examples,
                dialog,
                FindDialogueSummarizerParams(
                    key_episode=explore_payload.get("key_episode"),
                    identified_emotions=label_payload.get("identified_emotions"),
                ),
            )
            print(summarizer_result)
            if summarizer_result.proceed_to_next_phase is True and len(
                    current_state_ai_turns) >= 2:
                return EmotionChatbotPhase.Share, summarizer_result.model_dump()
            else:
                return None, summarizer_result.model_dump()
        # Share --> Explore or Terminate
        elif current == EmotionChatbotPhase.Share:
            last_turn_with_flag, last_turn_with_flag_index = dialogue_utils.find_last_turn(dialog, lambda turn: dict_utils.get_nested_value(turn.metadata, "new_episode_requested") == True)
            if last_turn_with_flag_index != -1 and last_turn_with_flag_index < len(dialog)-1:
                # if the flagged turn exists and is not the last one...
                result = await share.summarizer.run(
                    None,
                    dialog[last_turn_with_flag_index:],
                    FindDialogueSummarizerParams(
                        key_episode=explore_payload.get("key_episode"),
                        identified_emotions=label_payload.get("identified_emotions"),
                    ),
                )
                if result.share_new_episode:
                    return EmotionChatbotPhase.Explore, {"revisited": True}
        return None

    async def _get_response_impl(self, dialog: Dialogue, dry: bool = False) -> tuple[str, dict | None]:
        msg, metadata = await super()._get_response_impl(dialog, dry)
        return msg, dict_utils.set_nested_value(metadata, "locale", self.locale)

    @staticmethod
    def get_csv_writer(session_id: str)->DialogueCSVWriter:
        return DialogueCSVWriter(
            columns=["state", *[key for token, key, _ in SPECIAL_TOKEN_CONFIG], "model", "prompt_tokens", "message_tokens"],
            column_extractors=[
                TurnValueExtractor(["metadata", "state"]),
                *[TurnValueExtractor(["metadata", key]) for token, key, value in SPECIAL_TOKEN_CONFIG],
                TurnValueExtractor(["metadata", "chatgpt", "model"]),
                TurnValueExtractor(["metadata", "chatgpt", "usage", "prompt_tokens"]),
                TurnValueExtractor(["metadata", "chatgpt", "usage", "completion_tokens"])
            ]
        ).insertColumn("session", lambda turn, index, params: session_id, 0)
