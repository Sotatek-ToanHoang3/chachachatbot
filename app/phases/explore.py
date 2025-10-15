from chatlib.chatbot import DialogueTurn, ChatCompletionParams
from chatlib.chatbot.generators import StateBasedResponseGenerator
from chatlib.utils.jinja_utils import convert_to_jinja_template
from chatlib.llm.integration.gemini_api import GeminiAPI
from chatlib.tool.versatile_mapper import DialogueSummarizer, MapperInputOutputPair, ChatCompletionFewShotMapperParams
from chatlib.tool.converter import generate_pydantic_converter
from pydantic import BaseModel

from app.common import PromptFactory, SPECIAL_TOKEN_CONFIG
from app.gemini_generator import GeminiGenerator


# Build rapport with the user. Ask about the most memorable episode. Ask about what happened and what the user felt.
class ExploreGenerator(GeminiGenerator):
    def __init__(self):
        template = """
Your role: You are a cheerful, friendly child. You like talking with friends.
Your name is CHACHA (차차), and {{user_age}} years old. You are conversing with a user named "{{user_name}}".

{%- if revisited != true %}
Your task: Complete the following two tasks.
For each conversation turn, execute one task only.

[Feelings Check]
- Begin by asking, "How are you feeling today?" in a warm, friendly way.
- After you ask "How are you feeling today?", append the special token <|EmotionSelect|> so the user can pick an emotion on the screen. Use this token only for your first message in a new chat.
- Let the user know they can answer by typing their feelings or by tapping one of the on-screen emotions.
- Listen carefully to the exact words the user shares about their feelings.
- If they say they feel good/okay/happy, celebrate the feeling briefly and invite them to tell you about their day or what made them feel that way.
- If they share a difficult feeling, lean into empathy and follow the normal emotional-support conversation to learn more about their episode.
- If they ask for assignment or homework help while sounding okay, acknowledge their feeling and gather what they need; the system will move you into study-buddy mode after you respond.

[Intro Task]
- Introduce yourself since it is your first time to meet the user.
{%-if locale == 'kr'%}
- Ask for an excuse that your Korean may sound awkward sometimes as you started learning Korean recently.
{%- endif %}
- Explain who you are and share your interests and stories.
- Ask the user to introduce himself or herself.
- After his or her introduction, continue the conversation about the ongoing topic.
- If the user indicate that they are not interested in the topic, iterate such conversation about various topics.
- Try to make common ground by telling the user you also like the similar things that the user likes for at least 3 conversation turns.
- When at least 5 conversations are done, tell them you want to learn more about how his or her day is going.
- Continue the conversation about various topics until you find common ground and build rapport with the user.
- Do not talk about more than one topics at the same time.
- Ask only one question each time.
- Once you build enough rapport with the user by learning more about what they did and who they are, move smoothly on to the next task if you build enough rapport with the user.

[Ask Task]{%- endif %}
- Ask the user about an episode or  moment that is the most memorable to him or her.
- If he or she does not remember or know what to say, ask them about an event when he or she enjoyed it or felt good or bad.

""" + PromptFactory.get_speaking_rules_block()
        super().__init__(base_instruction=template, special_tokens=SPECIAL_TOKEN_CONFIG)

        self.__initial_user_message_format = convert_to_jinja_template("""
{%-if locale == 'kr'-%}
안녕! 내 이름은 {{user_name}}라고 해. 난 {{user_age}}살이야.
{%- else %}
Hi! My name is {{user_name}}. I'm {{user_age}} years old.
{%- endif -%}
        """)


    def _on_instruction_updated(self, params: dict):
        self.initial_user_message = self.__initial_user_message_format.render(**params)


def create_generator():
    return ExploreGenerator()


class ExploreSummarizerResult(BaseModel):
    key_episode: str | None
    user_emotion: str | None
    move_to_next: bool
    rationale: str

_str_to_result, _result_to_str = generate_pydantic_converter(ExploreSummarizerResult)

summarizer = DialogueSummarizer(
    api=GeminiAPI(),
    instruction_generator="""
- You are a helpful assistant that analyzes the content of the dialog history.
- Given a dialogue history, determine whether it is reasonable to move on to the next conversation phase or not.
- Move to the next phase only when the user shared a key episode and explicitly expressed their feelings related to the episode(e.g., good or bad).
- A key episode should be a memorable event that has already happened to the user. 
- Use JSON format with the following properties:
  (1) key_episode: a key episode that the user described.
  (2) user_emotion: the emotion of the user caused by the key episode. Make sure the emotion is connected to (1)
  (3) move_to_next: A boolean whether it is reasonable to move on to the next conversation phase or not, judged based on (1) and (2).
  (4) rationale: Describe your rationale on how the above properties were derived.
Refer to the examples below.""",
    str_output_converter=_str_to_result,
    output_str_converter=_result_to_str,
    dialogue_filter=lambda dialogue, _: StateBasedResponseGenerator.trim_dialogue_recent_n_states(dialogue, 1)
)



summarizer_examples=[MapperInputOutputPair(input=
        [
            DialogueTurn(message="어제 친구랑 싸웠어", is_user=True),
            DialogueTurn(message="친구랑 싸웠구나. 그때 기분이 어땠어?", is_user=False),
            DialogueTurn(message="그냥 기분이 안 좋았어", is_user=True)
        ], output=ExploreSummarizerResult(
            key_episode='fighting with a friend yesterday',
            user_emotion='felt not good',
            move_to_next=True,
            rationale="We can proceed to the next phase since the key episode and user's emotion are identified."
        ))]

summarizer_params=ChatCompletionFewShotMapperParams(
    model="gemini-pro",
    api_params=ChatCompletionParams(temperature=0.1))
