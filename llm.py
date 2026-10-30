# llm.py
import json
import logging
import config

try:
    from zai import ZhipuAiClient
except ImportError:
    ZhipuAiClient = None

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self):
        self.api_key = config.API_KEY
        self.model_name = config.MODEL_NAME
        self.client = None

        if ZhipuAiClient:
            try:
                self.client = ZhipuAiClient(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
        else:
            logger.warning("ZhipuAiClient not found; ignore this message if running in a test environment.")

    def call_model_sync(self, prompt: str) -> str:
        """Synchronous call to the LLM."""
        if not self.client:
            raise RuntimeError("Client not initialized. Check SDK installation and API key.")

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.1,  # low temperature for accurate structured extraction
                max_tokens=1024
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API call error: {e}")

    @staticmethod
    def parse_json_output(text: str) -> dict:
        """Robust JSON parser that handles Markdown fences the model may add."""
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: locate the first '{' and the last '}'
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            raise ValueError(f"Unable to parse JSON content: {text[:50]}...")
