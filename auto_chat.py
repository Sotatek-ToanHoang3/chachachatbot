import asyncio
from os import path, getcwd, getenv, environ

import openai
from chatlib.utils.validator import make_non_empty_string_validator
from chatlib.chatbot.generators import ChatGPTResponseGenerator
from chatlib.llm.integration.openai_api import ChatGPTModel
from dotenv import load_dotenv
from questionary import prompt

from app.common import ChatbotLocale
from app.response_generator import EmotionChatbotResponseGenerator
from chatlib.utils import cli
from chatlib.global_config import GlobalConfig

if __name__ == "__main__":
    # Init OpenAI API
    GlobalConfig.is_cli_mode = True
    load_dotenv(path.join(getcwd(), ".env"))
    if not getenv("GOOGLE_API_KEY") and getenv("GEMINI_API_KEY"):
        environ["GOOGLE_API_KEY"] = getenv("GEMINI_API_KEY")

    configuration_questions = [
        {
            'type': 'text',
            'name': 'user_name',
            'message': "Please enter child's name:",
            "validate": make_non_empty_string_validator("Please enter a valid name.")
        },
        {
            'type': 'text',
            'name': 'user_age',
            'message': "Please enter child's age:",
            "validate": lambda number: True if number.isnumeric() and int(
                number) >= 0 else "The age should be a number equal or larger than 0."
        },
        {
            'type': 'select',
            "name": 'locale',
            'choices': ['Korean', 'English'],
            'message': 'Select language you would like to speak in:'
        }
    ]

    configuration_answers = prompt(configuration_questions)

    user_name = configuration_answers['user_name'].strip()
    user_age = int(configuration_answers['user_age'].strip())
    locale_name = configuration_answers['locale'].strip()
    if locale_name == "Korean":
        locale = ChatbotLocale.Korean
    elif locale_name == "English":
        locale = ChatbotLocale.English
    else:
        raise ValueError("Unsupported locale.")

    asyncio.run(cli.run_auto_chat_loop(
        EmotionChatbotResponseGenerator(user_name=user_name, user_age=user_age, locale=locale, verbose=True),
        ChatGPTResponseGenerator(base_instruction=f"""
You are a {user_age}-year-old {locale_name} child named {user_name}.            
[Speaking Rules]
{"1. Use a simple, informal Korean like talking to a peer friend." if locale is ChatbotLocale.Korean else "1. Use a simple, informal English like talking to a peer friend."}
2. You should be saying like {user_age}-year-old; be brief and short within two sentences.  
3. Do not ask question unless you do not understand certain emotion words.
4. When asked about the day, say that you had a bit of hard time today, becasue your mom had you do your math homework by yourself. But you're not really good at math, so you got in trouble with her.
5. You are struggling with naming your emotion regarding the case.
""", model=ChatGPTModel.GPT_4o),
        max_turns=30
    ))
