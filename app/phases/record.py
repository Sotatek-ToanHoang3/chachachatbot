import json

from chatlib.chatbot import DialogueTurn, RegenerateRequestException
from chatlib.chatbot.generators import StateBasedResponseGenerator
from app.gemini_generator import GeminiGenerator
from chatlib.utils.jinja_utils import convert_to_jinja_template
from chatlib.tool.versatile_mapper import DialogueSummarizer, Dialogue, DialogueTurn, MapperInputOutputPair
from chatlib.tool.converter import generate_pydantic_converter
from chatlib.llm.integration.gemini_api import GeminiAPI

from app.common import FindDialogueSummarizerParams, PromptFactory, SPECIAL_TOKEN_CONFIG, RecordSummarizerResult


# Encourage the user to record the moments in which they felt positive emotions.
def create_generator():
    template = (
        f"{PromptFactory.GENERATOR_PROMPT_BLOCK_KEY_EPISODE_AND_EMOTION_TYPES}\n"
        "\n"
        "- The goal of the current conversation is to encourage the user to keep diary to record the moments in which they felt positive emotions:\n"
        "{%- for em in identified_emotions | selectattr(\"is_positive\", \"true\") %}\n"
        "  * {{em.emotion}} ({{em.reason}})\n"
        "{%- endfor %}\n"
        "\n"
        "- 1. First start with asking the user whether they have been keeping diaries or journals regularly.\n"
        "- 2. Then encourage the user to keep diary to record the moments in which they felt positive emotions.\n"
        "- 3. Suggest a diary content by explicitly providing an example essay summarizing the above positive emotions and the reason;\n"
        "  {%- if locale == 'kr' %}for the essay, use '~다' style Korean, such as \"~했다.\" rather than \"~했어.\";\n{%- endif %}\n"
        "  put the diary content wrapped with <diary></diary>, at the end of the message;\n"
        "  {%- if locale == 'kr' %}use the phrase like \"예를 들어 다음과 같은 내용으로 일기를 써볼 수 있을 거야.\"{%- endif %}\n"
        "\n"
        "- Since the user is currently conversing with you, don't ask them to record now.\n"
    )
    guide_template = """
[Guide to the conversation]
- If not already asked, ask whether the user is keeping diary these days.
- If not already explained, explain the importance of recording the emotions.
- If not already provided, provide an example diary content.
"""
    template = template + guide_template + PromptFactory.get_speaking_rules_block()
    return GeminiGenerator(base_instruction=template, special_tokens=SPECIAL_TOKEN_CONFIG)


_summarizer_instruction_template = convert_to_jinja_template("""
- You are a helpful assistant that analyzes the content of the dialogue history.
""" + PromptFactory.SUMMARIZER_PROMPT_BLOCK_KEY_EPISODE_AND_EMOTION_TYPES + """
- The AI in the dialogue is encouraging the user to record the moments in which they felt positive emotions: {{ identified_emotions | selectattr("is_positive", "true") | map(attribute="emotion") | join(", ") }}.

- Analyze the input dialogue and identify if the AI had sufficient conversation about the recording.
Follow this JSON format: {
    "asked_user_keeping_diary": boolean, // true if the AI had asked whether the user is keeping diary at present.
    "explained_importance_of_recording": boolean, // true if the AI had described the importance of recording positive moments.
    "reflection_note_content_provided": boolean // Whether the AI has provided the reflection note to the user with <diary> tag.
}
""")

def _instruction_generator(dialogue: Dialogue, params: FindDialogueSummarizerParams)->str:
    return _summarizer_instruction_template.render(key_episode=params.key_episode, identified_emotions=params.identified_emotions)

_str_to_result, _result_to_str = generate_pydantic_converter(RecordSummarizerResult)

def _str_to_result_func(model_output: str, params: FindDialogueSummarizerParams) -> RecordSummarizerResult:
    try:
        result = _str_to_result(model_output, params)
        result.proceed_to_next_phase = result.asked_user_keeping_diary == True and result.explained_importance_of_recording == True and result.reflection_note_content_provided == True
        return result
    except:
        raise RegenerateRequestException("Malformed data.")


summarizer = DialogueSummarizer[RecordSummarizerResult, FindDialogueSummarizerParams](
    api=GeminiAPI(),
    instruction_generator=_instruction_generator,
    dialogue_filter=lambda dialogue, _: StateBasedResponseGenerator.trim_dialogue_recent_n_states(
                             dialogue, 3),
    output_str_converter=_result_to_str,
    str_output_converter=_str_to_result_func
)
     

summarizer_examples=[MapperInputOutputPair(input=[
                        DialogueTurn(message="오늘 좋았던 기분을 일기에 써보는건 어때?", is_user=False),
                        DialogueTurn(message="뭐라고 써야 할지 모르겠어", is_user=True),
                        DialogueTurn(message="이런식으로 써도 좋을 것 같아! <diary>오늘은 정말 감동적인 하루였다. 친구들과 축구를 했는데, 내가 역전골을 넣어서 정말 신났다.</diary>", is_user=False),
                    ], output= RecordSummarizerResult(
                        asked_user_keeping_diary=False,
                        explained_importance_of_recording= False,
                        reflection_note_content_provided= True)),
                    MapperInputOutputPair(input=[
                        DialogueTurn(message="응. 오늘 오랜만에 친구들을 만나서 행복했어", is_user=True),
                        DialogueTurn(message="그랬구나. 윤수는 혹시 일기같은 걸 써?", is_user=False),
                        DialogueTurn(message="근데 난 일기 같은거 안써", is_user=True),
                        DialogueTurn(message="오늘 행복했던 기분을 일기에 써보는 건 어때? 일기 쓰는 건 처음에는 좀 어색할 수 있지만, 시간이 지날수록 이런 감정들을 기록하고 되돌아보는 게 재미있단다.", is_user=False)
                    ], output= RecordSummarizerResult(
                        asked_user_keeping_diary=True,
                        explained_importance_of_recording=True,
                        reflection_note_content_provided=False,
                    ))
        ]