import google.generativeai as genai
from typing import List, Dict, Any, Optional

class GeminiAPI:
    def __init__(self):
        self.api_key = None
        self._model = None
        
    @property
    def api_key(self) -> Optional[str]:
        return self._api_key
        
    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value
        if value:
            genai.configure(api_key=value)
            self._model = genai.GenerativeModel('gemini-pro')

    async def generate_content(
        self,
        contents: List[Dict[str, Any]],
        generation_config: Dict[str, Any] = None
    ) -> Any:
        if not self._model:
            raise ValueError("API key not set. Please set api_key before making requests.")
            
        # Convert messages format to Gemini's expected format
        chat = self._model.start_chat()
        history = []
        
        for msg in contents:
            if msg["role"] == "system":
                # For system messages, we'll store them to prepend to user messages
                system_prompt = msg["parts"][0]["text"]
                continue
                
            if msg["role"] == "user":
                # For user messages, prepend system prompt if it exists
                content = msg["parts"][0]["text"]
                if system_prompt and len(history) == 0:
                    content = f"{system_prompt}\n\nUser: {content}"
                response = await chat.send_message_async(
                    content,
                    generation_config=generation_config
                )
                history.append(response)
            
        # Return the last response if it exists, otherwise return empty response
        if history:
            return history[-1]
        return type('Response', (), {
            'candidates': [
                type('Candidate', (), {
                    'content': type('Content', (), {
                        'parts': [
                            type('Part', (), {'text': ''})()
                        ]
                    })()
                })()
            ]
        })()