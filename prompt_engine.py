# ai_service.py
import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# --- Configuration ---
# IMPORTANT: Use environment variables for API keys in a real project!
# For this example, we'll hardcode it for simplicity.
#
# MAKE SURE TO REPLACE THIS WITH YOUR ACTUAL GOOGLE API KEY
# <--- IMPORTANT: PASTE YOUR KEY HERE
GOOGLE_API_KEY = 'ADD_UR_API_HERE'
# This is the correct address for your local server
LM_STUDIO_API_URL = 'http://127.0.0.1:1234/v1/chat/completions'

# --- Initialize Google AI ---
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # CORRECTED: Use the stable 'gemini-1.0-pro' model name
    google_model = genai.GenerativeModel('gemini-2.5-flash')
    print("Successfully configured Google AI.")
except Exception as e:
    print(
        f"CRITICAL ERROR: Could not configure Google AI. Check your API key. Error: {e}")
    google_model = None

# --- Helper Function for LM Studio (Now configured for your Gemma model) ---


def enhance_prompt_with_local_llm(user_prompt):
    """Sends the prompt to your local Gemma model for enhancement."""
    meta_prompt = f"""
    You are a prompt-enhancing assistant. Your job is to rewrite the user's prompt to be more detailed,
    clear, and effective for a powerful AI model. Return only the enhanced prompt itself, without any extra phrases like "Here is the enhanced prompt:".

    Original Prompt: "{user_prompt}"
    """
    # This payload now specifies your model's identifier.
    payload = {
        "model": "google/gemma-3-4b",  # <-- Your specific model identifier
        "messages": [{"role": "user", "content": meta_prompt}],
        "temperature": 0.4,
        # "max_tokens": 150 # Optional: You can add this to limit the length of the enhanced prompt
    }
    print(
        f"--- Sending to local model ({payload['model']}) for enhancement... ---")
    try:
        response = requests.post(LM_STUDIO_API_URL, json=payload)
        # Raise an exception for bad status codes (like 404 or 500)
        response.raise_for_status()
        enhanced_prompt = response.json(
        )['choices'][0]['message']['content'].strip()
        print(f"--- Enhanced Prompt Received: {enhanced_prompt} ---")
        return enhanced_prompt
    except requests.exceptions.RequestException as e:
        print(f"Error calling local LM Studio model: {e}")
        print("Falling back to the original prompt.")
        return user_prompt  # Fallback to the original prompt on error

# --- Main API Route ---


@app.route('/chat', methods=['POST'])
def chat_handler():
    """
    Handles chat requests, optionally enhances the prompt with a local LLM,
    and gets a response from Google's Gemini model.
    Includes robust error handling for blocked responses.
    """
    if not google_model:
        return jsonify({"error": "Google AI model is not configured. Check server logs."}), 500

    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is missing"}), 400

    user_prompt = data['prompt']
    # The 'enhance' flag defaults to False if not provided
    enhance = data.get('enhance', False)

    final_prompt = user_prompt
    if enhance:
        # This function call now uses your Gemma model
        final_prompt = enhance_prompt_with_local_llm(user_prompt)

    try:
        # --- Call Google AI Studio ---
        print(f"--- Sending final prompt to Google AI: {final_prompt} ---")
        response = google_model.generate_content(final_prompt)

        # --- ROBUSTNESS FIX: Check for content before accessing it ---
        # The API can return an empty response if content is blocked by safety filters.
        if response.parts:
            return jsonify({"response": response.text})
        else:
            # The response was likely blocked by safety settings.
            print("--- Google AI returned an empty response (likely blocked) ---")
            return jsonify({"error": "The response was blocked due to safety concerns. Please try a different prompt."}), 400

    except Exception as e:
        # This will catch other errors, like network issues or API misconfigurations.
        print(f"Error calling Google AI: {e}")
        return jsonify({"error": "Failed to get response from Google AI", "details": str(e)}), 500


if __name__ == '__main__':
    # Runs the AI service on port 5001
    app.run(port=5001, debug=True)
