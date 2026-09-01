# universal_ai_connector.py
# OpenAI / Gemini / Claude / Mistral / HuggingFace / Groq / Local など多数AI API対応の統合AIコネクタ

import os

import requests
from dotenv import load_dotenv


class UniversalAIConnector:
    def __init__(self, backend=None, model=None):
        load_dotenv()

        supported_backends = {
            "openai": self._init_openai,
            "gemini": self._init_gemini,
            "claude": self._init_claude,
            "mistral": self._init_mistral,
            "huggingface": self._init_huggingface,
            "groq": self._init_groq,
            "local": self._init_local
        }

        while True:
            if not backend:
                backend = input("使用したいAIバックエンドを入力してください（openai / gemini / claude / mistral / huggingface / groq / local）: ").strip().lower()

            if backend in supported_backends:
                supported_backends[backend](model)
                break
            else:
                print(f"'{backend}' は未対応です。もう一度入力してください。")
                backend = None

        self.backend = backend

    def _init_openai(self, model):
        import openai
        while True:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY が見つかりません。以下に入力してください。")
                api_key = input("あなたの OpenAI APIキーを入力: ").strip()
                if api_key.startswith("sk-"):
                    with open(".env", "a") as f:
                        f.write(f"\nOPENAI_API_KEY={api_key}")
                    os.environ["OPENAI_API_KEY"] = api_key
                    break
            else:
                break
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model or "gpt-3.5-turbo"

    def _init_gemini(self, model):
        import google.generativeai as genai
        while True:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("GEMINI_API_KEY が見つかりません。以下に入力してください。")
                api_key = input("あなたの Gemini APIキーを入力: ").strip()
                if api_key:
                    with open(".env", "a") as f:
                        f.write(f"\nGEMINI_API_KEY={api_key}")
                    os.environ["GEMINI_API_KEY"] = api_key
                    break
            else:
                break
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(model or "gemini-pro")
        self.model = None

    def _init_claude(self, model):
        import anthropic
        while True:
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key:
                print("CLAUDE_API_KEY が見つかりません。以下に入力してください。")
                api_key = input("あなたの Claude APIキーを入力: ").strip()
                if api_key:
                    with open(".env", "a") as f:
                        f.write(f"\nCLAUDE_API_KEY={api_key}")
                    os.environ["CLAUDE_API_KEY"] = api_key
                    break
            else:
                break
        self.client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        self.model = model or "claude-3-opus-20240229"

    def _init_mistral(self, model):
        while True:
            api_key = os.getenv("MISTRAL_API_KEY")
            if not api_key:
                print("MISTRAL_API_KEY が見つかりません。以下に入力してください。")
                api_key = input("あなたの Mistral APIキーを入力: ").strip()
                if api_key:
                    with open(".env", "a") as f:
                        f.write(f"\nMISTRAL_API_KEY={api_key}")
                    os.environ["MISTRAL_API_KEY"] = api_key
                    break
            else:
                break
        self.model = model or "mistral-medium"

    def _init_huggingface(self, model):
        while True:
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                print("HUGGINGFACE_API_KEY が見つかりません。以下に入力してください。")
                api_key = input("あなたの HuggingFace APIキーを入力: ").strip()
                if api_key:
                    with open(".env", "a") as f:
                        f.write(f"\nHUGGINGFACE_API_KEY={api_key}")
                    os.environ["HUGGINGFACE_API_KEY"] = api_key
                    break
            else:
                break
        self.model = model or "HuggingFaceH4/zephyr-7b-beta"

    def _init_groq(self, model):
        while True:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("GROQ_API_KEY が見つかりません。以下に入力してください。")
                api_key = input("あなたの Groq APIキーを入力: ").strip()
                if api_key:
                    with open(".env", "a") as f:
                        f.write(f"\nGROQ_API_KEY={api_key}")
                    os.environ["GROQ_API_KEY"] = api_key
                    break
            else:
                break
        self.model = model or "llama3-8b-8192"

    def _init_local(self, model):
        print("ローカルAIを使用します（例: llama.cpp）")
        self.client = None
        self.model = model or "local-llama"

    def ask(self, prompt: str) -> str:
        dispatch = {
            "openai": self._ask_openai,
            "gemini": self._ask_gemini,
            "claude": self._ask_claude,
            "mistral": self._ask_mistral,
            "huggingface": self._ask_huggingface,
            "groq": self._ask_groq,
            "local": self._ask_local
        }
        return dispatch.get(self.backend, self._ask_unknown)(prompt)

    def _ask_openai(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a structure-aware assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def _ask_gemini(self, prompt):
        response = self.client.generate_content(prompt)
        return response.text.strip()

    def _ask_claude(self, prompt):
        with self.client.messages.stream(model=self.model, max_tokens=1024) as stream:
            stream.send_message(prompt)
            return stream.get_final_message().content[0].text.strip()

    def _ask_mistral(self, prompt):
        headers = {
            "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        res = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
        return res.json()["choices"][0]["message"]["content"].strip()

    def _ask_huggingface(self, prompt):
        headers = {
            "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"return_full_text": False}
        }
        res = requests.post(f"https://api-inference.huggingface.co/models/{self.model}", headers=headers, json=payload)
        return res.json()[0]["generated_text"]

    def _ask_groq(self, prompt):
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        return res.json()["choices"][0]["message"]["content"].strip()

    def _ask_local(self, prompt):
        return f"[ローカルAI仮想応答]：'{prompt}' に対してローカルモデルが応答します（未実装）。"

    def _ask_unknown(self, prompt):
        return f"[{self.backend.upper()}仮想応答]：'{prompt}' に対してはまだAPI統合が実装されていません。ごめんね！"

if __name__ == "__main__":
    ai = UniversalAIConnector()
    reply = ai.ask("構造知に基づく次の行動を教えて")
    print("AI Response:", reply)
