# gemini_connector.py
# Google Gemini APIと連携して、構造知や状態に対する解釈・応答を得るモジュール

import os
from dotenv import load_dotenv
import google.generativeai as genai

class GeminiConnector:
    def __init__(self, model="gemini-pro"):
        load_dotenv()  # .envファイルの読み込み
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY が見つかりません。以下に入力してください。")
            api_key = input("あなたの Gemini APIキーを入力: ").strip()
            if not api_key:
                raise ValueError("APIキーが空です")
            with open(".env", "a") as f:
                f.write(f"\nGEMINI_API_KEY={api_key}")
            print(".env に APIキーを保存しました。次回からは自動で読み込まれます。")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def ask(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()

if __name__ == "__main__":
    gemini = GeminiConnector()
    result = gemini.ask("空気感が崩壊したときに、どんな振る舞いが望ましいですか？")
    print("Gemini Response:", result)
