from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import base64
import requests
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
google_api_key = os.getenv('GOOGLE_API_KEY')
deepai_api_key = os.getenv('DEEPAI_API_KEY')
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
    """Generate image using DeepAI (free tier)"""
    try:
        key = deepai_api_key
        if not key:
            return {'success': False, 'error': 'DEEPAI_API_KEY not configured in environment'}

        r = requests.post(
            'https://api.deepai.org/api/text2img',
            data={'text': prompt},
            headers={'api-key': key},
            timeout=60
        )

        if r.status_code == 200:
            result = r.json()
            image_url = result.get('output_url')
            if not image_url:
                return {'success': False, 'error': 'No image URL in response'}

            # Fetch the image and convert to base64
            img_response = requests.get(image_url, timeout=30)
            if img_response.status_code == 200:
                b64 = base64.b64encode(img_response.content).decode('utf-8')
                return {'success': True, 'image': f'data:image/jpeg;base64,{b64}'}
            else:
                return {'success': False, 'error': f'Failed to fetch image: HTTP {img_response.status_code}'}
        else:
            return {'success': False, 'error': f'DeepAI returned HTTP {r.status_code}: {r.text[:200]}'}

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
        'imageConfigured': bool(deepai_api_key)
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 10000))

    print('\nMoonAI')
    print('━' * 42)
    print(f'✅ Server running on port {port}')
    print(f'🌐 Frontend: http://localhost:{port}')
    print(f'📡 Health check: http://localhost:{port}/api/health')
    print(f'🔑 Gemini API: {"Configured ✓" if google_api_key else "Not configured ✗"}')
    print(f'🎨 Image API: {"DeepAI ✓" if deepai_api_key else "Not configured ✗"}')
    print('━' * 42 + '\n')

    app.run(host='0.0.0.0', port=port, debug=False)