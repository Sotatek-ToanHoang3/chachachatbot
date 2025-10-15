from os import path, getcwd

from chatlib.global_config import GlobalConfig
from chatlib.llm.integration.gemini_api import GeminiAPI
from dotenv import find_dotenv, set_key, load_dotenv

#Create env files if not exist ========================================================

if __name__ == "__main__":
    GlobalConfig.is_cli_mode = True
    api = GeminiAPI()
    api.api_key = "AIzaSyBjqYCP4X4C1Sl43IWdrUmQMOYMwH728-c"
    with open(".env", "w") as f:
        f.write(f'GEMINI_API_KEY="{api.api_key}"')

print("Wrote .env file with Gemini API key.")