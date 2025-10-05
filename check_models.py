# check_models.py

import os
import google.generativeai as genai

# rag_test.py と同じAPIキーを設定
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8KFC53EjU9YKti7nGgeYQi6wq3BWSxxw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("利用可能なモデル:")
for m in genai.list_models():
  # 質問応答に使えるモデル（generateContentをサポート）のみ表示
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)