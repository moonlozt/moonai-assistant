from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure APIs
google_api_key = os.getenv('GOOGLE_API_KEY')
groq_api_key   = os.getenv('GROQ_API_KEY')

gemini_client = genai.Client(api_key=google_api_key) if google_api_key else None
groq_client   = Groq(api_key=groq_api_key) if groq_api_key else None

# In-memory conversation storage
conversations     = {}
conversation_turn = {}  # True = Gemini, False = Groq

# Keywords that indicate image generation requests
IMAGE_KEYWORDS = [
    'generate image', 'generate a image', 'generate an image',
    'create image', 'create a image', 'create an image',
    'make image', 'make a image', 'make an image',
    'draw', 'draw me', 'paint', 'paint me',
    'image of', 'picture of', 'photo of',
    'generate picture', 'create picture', 'make picture',
    'show me a', 'show me an',
]

# Keywords that indicate creator/about questions
CREATOR_KEYWORDS = [
    'who made you', 'who created you', 'who built you', 'who designed you',
    'who is your creator', 'who is your developer', 'who is your maker',
    'your creator', 'your developer', 'your maker', 'your author',
    'who owns you', 'who wrote you', 'who programmed you',
    'what are you', 'who are you', 'tell me about yourself',
    'about you', 'about yourself',
]


def is_image_request(message):
    lower = message.lower()
    return any(keyword in lower for keyword in IMAGE_KEYWORDS)


def is_creator_request(message):
    lower = message.lower()
    return any(keyword in lower for keyword in CREATOR_KEYWORDS)


def call_gemini_api(messages, api_key=None):
    """Call Google Gemini API"""
    try:
        key = api_key or google_api_key
        if not key:
            return {'success': False, 'error': 'No Gemini API key configured'}

        client = genai.Client(api_key=key)

        full_prompt = ''
        if len(messages) > 1:
            context = '\n'.join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in messages[:-1]
            ])
            full_prompt = f"Previous conversation:\n{context}\n\n"

        user_message = messages[-1]['content']
        full_prompt += f"User: {user_message}"

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_k=40,
                top_p=0.95,
                max_output_tokens=8192,
            )
        )

        return {'success': True, 'message': response.text, 'provider': 'Gemini'}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def call_groq_api(messages):
    """Call Groq API"""
    try:
        if not groq_client:
            return {'success': False, 'error': 'No Groq API key configured'}

        groq_messages = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in messages
        ]

        response = groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=groq_messages,
            temperature=0.7,
            max_tokens=8192,
        )

        return {'success': True, 'message': response.choices[0].message.content, 'provider': 'Groq'}

    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.route('/')
def index():
    return send_from_directory('.', 'moonai.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data            = request.json
        message         = data.get('message')
        api_key         = data.get('apiKey')
        conversation_id = data.get('conversationId', str(int(datetime.now().timestamp() * 1000)))

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Block image generation requests
        if is_image_request(message):
            return jsonify({
                'message': '🚫 **Image generation is not available.**\n\nI can only assist with text-based conversations. Try asking me something else!',
                'conversationId': conversation_id,
                'imageBlocked': True
            })

        # Creator/about questions
        if is_creator_request(message):
            return jsonify({
                'message': '🌙 I was created by **Moonlost**.',
                'conversationId': conversation_id
            })

        history = conversations.get(conversation_id, [])
        history.append({'role': 'user', 'content': message})
        messages = history[-20:]

        # Alternate between Gemini and Groq
        use_gemini = conversation_turn.get(conversation_id, True)
        conversation_turn[conversation_id] = not use_gemini

        if use_gemini:
            result = call_gemini_api(messages, api_key)
            if not result['success']:
                result = call_groq_api(messages)  # fallback
        else:
            result = call_groq_api(messages)
            if not result['success']:
                result = call_gemini_api(messages, api_key)  # fallback

        if not result['success']:
            return jsonify({'error': result['error']}), 500

        history.append({'role': 'assistant', 'content': result['message']})
        conversations[conversation_id] = history

        return jsonify({
            'message': result['message'],
            'conversationId': conversation_id,
            'provider': result.get('provider', 'AI')
        })

    except Exception as e:
        print(f'Chat error: {e}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    history = conversations.get(conversation_id)
    if not history:
        return jsonify({'error': 'Conversation not found'}), 404
    return jsonify({'history': history})


@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    if conversation_id in conversations:
        del conversations[conversation_id]
    if conversation_id in conversation_turn:
        del conversation_turn[conversation_id]
    return jsonify({'message': 'Conversation cleared'})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'geminiConfigured': bool(google_api_key),
        'groqConfigured':   bool(groq_api_key),
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))

    print('\nMoonAI')
    print('━' * 42)
    print(f'✅ Server running on port {port}')
    print(f'🌐 Frontend: http://localhost:{port}')
    print(f'📡 Health check: http://localhost:{port}/api/health')
    print(f'🔑 Gemini API: {"Configured ✓" if google_api_key else "Not configured ✗"}')
    print(f'⚡ Groq API:   {"Configured ✓" if groq_api_key else "Not configured ✗"}')
    print('━' * 42 + '\n')

    app.run(host='0.0.0.0', port=port, debug=False)