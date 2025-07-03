import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize the Gemini model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

def get_answer_from_context(question, context):
    prompt = f"Given the following legal context:\n{context}\n\nAnswer this question: {question}"
    response = model.generate_content(prompt)
    return {"answer": response.text.strip()}

def summarize_context(context):
    prompt = f"Summarize the following legal content concisely:\n{context}"
    response = model.generate_content(prompt)
    return {"answer": response.text.strip()}
