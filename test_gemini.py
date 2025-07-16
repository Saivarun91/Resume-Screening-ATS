import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print("KEY:", api_key[:10], "...")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-1.5-flash")

try:
    response = model.generate_content("Say hello in one word")
    print(response.text)
except Exception as e:
    print("Gemini API call failed:", str(e))
