from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import random
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from google import genai
from google.genai import types
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure APIs
google_api_key   = os.getenv('GOOGLE_API_KEY')
pollinations_key = os.getenv('POLLINATIONS_API_KEY')  # optional but recommended
client = genai.Client(api_key=google_api_key) if google_api_key else None

# In-memory conversation storage
conversations = {}


def call_gemini_api(messages, api_key=None):
    """Call Google Gemini API"""
    try:
        key = api_key or google_api_key
        if not key:
            return {'success': False, 'error': 'No API key configured'}

        api_client = genai.Client(api_key=key)

        full_prompt = ''
        if len(messages) > 1:
            context = '\n'.join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in messages[:-1]
            ])
            full_prompt = f"Previous conversation:\n{context}\n\n"

        user_message = messages[-1]['content']
        full_prompt += f"User: {user_message}"

        response = api_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_k=40,
                top_p=0.95,
                max_output_tokens=8192,
            )
        )

        return {'success': True, 'message': response.text}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def generate_image(prompt, api_key=None):
    """Generate image using Pollinations AI (free, no SDK needed)"""
    try:
        seed = random.randint(1, 999999)
        encoded = quote(prompt)

        # Build URL — image.pollinations.ai is the correct endpoint
        url = (
            f'https://image.pollinations.ai/prompt/{encoded}'
            f'?model=flux&width=1024&height=1024&seed={seed}&nologo=true'
        )

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://pollinations.ai/',
        }

        # Add token if available (get free one at auth.pollinations.ai)
        if pollinations_key:
            headers['Authorization'] = f'Bearer {pollinations_key}'

        r = requests.get(url, headers=headers, timeout=90)

        print(f'Pollinations status: {r.status_code}, content-type: {r.headers.get("content-type")}')

        if r.status_code == 200 and 'image' in r.headers.get('content-type', ''):
            b64 = base64.b64encode(r.content).decode('utf-8')
            return {'success': True, 'image': f'data:image/jpeg;base64,{b64}'}
        else:
            return {'success': False, 'error': f'Pollinations returned HTTP {r.status_code}: {r.text[:200]}'}

    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Image generation timed out. Please try again.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.route('/')
def index():
    return send_from_directory('.', 'moonai.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        api_key = data.get('apiKey')
        conversation_id = data.get('conversationId', str(int(datetime.now().timestamp() * 1000)))

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        history = conversations.get(conversation_id, [])
        history.append({'role': 'user', 'content': message})
        messages = history[-20:]

        result = call_gemini_api(messages, api_key)
        if not result['success']:
            return jsonify({'error': result['error']}), 500

        history.append({'role': 'assistant', 'content': result['message']})
        conversations[conversation_id] = history

        return jsonify({'message': result['message'], 'conversationId': conversation_id})

    except Exception as e:
        print(f'Chat error: {e}')
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/api/generate-image', methods=['POST'])
def generate_image_endpoint():
    try:
        data = request.json
        prompt = data.get('prompt')
        api_key = data.get('apiKey')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        result = generate_image(prompt, api_key)
        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({'image': result['image']})

    except Exception as e:
        print(f'Image generation error: {e}')
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
    return jsonify({'message': 'Conversation cleared'})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'geminiConfigured': bool(google_api_key),
        'pollinationsToken': 'yes' if pollinations_key else 'no (anonymous, may be slow)'
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))

    print('\nMoonAI')
    print('━' * 42)
    print(f'✅ Server running on port {port}')
    print(f'🌐 Frontend: http://localhost:{port}')
    print(f'📡 Health check: http://localhost:{port}/api/health')
    print('━' * 42 + '\n')

    app.run(host='0.0.0.0', port=port, debug=False)