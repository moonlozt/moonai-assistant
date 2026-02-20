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

# Keywords that indicate code requests (checked before image to avoid false positives)
CODE_KEYWORDS = [
    'code', 'syntax', 'script', 'program', 'function',
    'example', 'snippet', 'how to code', 'how to write',
    'show me how', 'show me a example', 'show me a syntax',
    'show me a code', 'show me a script', 'show me a function',
    'write a', 'write me a', 'write me the',
    'give me a code', 'give me a syntax', 'give me a script',
    'give me an example', 'give me a function',
    'print hello', 'hello world', 'algorithm', 'pseudocode',
    'how do i', 'how to make a', 'how to build a',
    'tutorial', 'explain how', 'show me how to',
    'loop', 'array', 'string', 'integer', 'variable',
    'class', 'object', 'method', 'import', 'library',
]

# Keywords that indicate image generation requests
IMAGE_KEYWORDS = [
    # /command style
    '/image', '/img', '/imagine', '/draw', '/paint', '/generate',
    # generate
    'generate image', 'generate a image', 'generate an image',
    'generate me', 'generate a photo', 'generate a picture',
    'generate a drawing', 'generate a illustration',
    # create
    'create image', 'create a image', 'create an image',
    'create me', 'create a photo', 'create a picture',
    'create a drawing', 'create a illustration',
    # make
    'make image', 'make a image', 'make an image',
    'make me', 'make a photo', 'make a picture',
    'make a drawing', 'make a illustration',
    # draw
    'draw', 'draw me', 'draw a', 'draw an', 'draw something',
    # paint
    'paint', 'paint me', 'paint a', 'paint an',
    # show
    'show me a', 'show me an', 'show me image', 'show me picture',
    # render
    'render', 'render me', 'render a', 'render an',
    # image/picture/photo of
    'image of', 'picture of', 'photo of', 'illustration of',
    'drawing of', 'painting of', 'sketch of',
    # generate picture/photo
    'generate picture', 'create picture', 'make picture',
    'generate photo', 'create photo', 'make photo',
    # can you
    'can you draw', 'can you paint', 'can you generate',
    'can you create a image', 'can you make a image',
    'can you create an image', 'can you make an image',
    'can you show me a picture', 'can you show me an image',
    # misc
    'visualize', 'visualise', 'design me', 'design a',
    'produce an image', 'produce a picture',
]

# Keywords that indicate creator/about questions
CREATOR_KEYWORDS = [
    # who made/created/built
    'who made you', 'who made u', 'who made this',
    'who created you', 'who created u', 'who created this',
    'who built you', 'who built u', 'who built this',
    'who designed you', 'who designed u', 'who designed this',
    'who developed you', 'who developed u', 'who developed this',
    'who coded you', 'who coded u', 'who coded this',
    'who programmed you', 'who programmed u', 'who programmed this',
    'who wrote you', 'who wrote u', 'who made u bro',
    # your creator/owner
    'who is your creator', 'who is your developer',
    'who is your maker', 'who is your owner',
    'who is your author', 'who is your programmer',
    'your creator', 'your developer', 'your maker',
    'your author', 'your owner', 'your programmer',
    # who are you / what are you
    'who are you', 'who r u', 'who ru',
    'what are you', 'what r u',
    'what is moonai', 'what is moon ai',
    'tell me about yourself', 'tell me about you',
    'about you', 'about yourself',
    # introduce yourself
    'introduce yourself', 'introduce urself',
    # are you
    'are you ai', 'are you a bot', 'are you a robot',
    'are you chatgpt', 'are you gemini', 'are you gpt',
    # made by
    'made by who', 'created by who', 'built by who',
    'developed by who', 'coded by who',
    # origin
    'where are you from', 'where do you come from',
    'what is your origin', 'whats your origin',
]

# Keywords asking about moonlost
MOONLOST_KEYWORDS = [
    'who is moonlost', 'who is moon lost', 'moonlost who',
    'tell me about moonlost', 'tell me about moon lost',
    'what is moonlost', 'what is moon lost',
    'moonlost info', 'moonlost information',
    'who is the owner', 'who is the creator moonlost',
    'about moonlost', 'moonlost?',
    'moonlost developer', 'moonlost creator',
    'is moonlost a person', 'is moonlost real',
    'who is moon',
]

# ── Edit this to say whatever you want about moonlost ──────────────────────
MOONLOST_RESPONSE = """**Moonlost🌕** — Creator & Developer of MoonAI

Despite the rumors of his otherworldly origins, Moonlost is not an alien; he is a human visionary who looked at the stars and decided to build a bridge for the rest of us. He is the founder and commanding leader of **GoS**, the galaxy's most prestigious collective of space explorers dedicated to charting the unknown reaches of the deep cosmos.

Moonlost engineered the **MoonAI** structure with a singular goal: to streamline human labor through advanced technology, making our daily lives as effortless as a walk in zero gravity. But even a cosmic pioneer needs a break. When he isn't leading interstellar expeditions or refining neural networks, Moonlost enjoys the simple joys of life — immersing himself in video games and relaxing with his loyal companion, **Luki🐶**, a pet dog who has traveled more light-years✨ than most humans can imagine."""
# ───────────────────────────────────────────────────────────────────────────


def is_code_request(message):
    lower = message.lower().strip()
    return any(keyword in lower for keyword in CODE_KEYWORDS)


def is_image_request(message):
    lower = message.lower().strip()
    return any(keyword in lower for keyword in IMAGE_KEYWORDS)


def is_creator_request(message):
    lower = message.lower().strip()
    return any(keyword in lower for keyword in CREATOR_KEYWORDS)


def is_moonlost_request(message):
    lower = message.lower().strip()
    return any(keyword in lower for keyword in MOONLOST_KEYWORDS)


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


@app.after_request
def add_language_header(response):
    response.headers['Content-Language'] = 'en'
    return response


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

        # Block image generation requests (skip if it's actually a code request)
        if is_image_request(message) and not is_code_request(message):
            return jsonify({
                'message': '🚫 **Image generation is not available.**\n\nI can only assist with text-based conversations. Try asking me something else!',
                'conversationId': conversation_id,
                'imageBlocked': True
            })

        # Moonlost questions (check before creator so it's more specific)
        if is_moonlost_request(message):
            return jsonify({
                'message': MOONLOST_RESPONSE,
                'conversationId': conversation_id
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