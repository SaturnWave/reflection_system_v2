from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import PyPDF2
import docx
import io
import functools

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure secret key for session management

# Hardcoded access code (in production, store securely)
ACCESS_CODE = "111111111"

# Initialize Fireworks API client for AI generation
fireworks_client = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)
FIREWORKS_MODEL_ID = os.getenv("FIREWORKS_MODEL_ID", "accounts/fireworks/models/llama-v3p1-70b-instruct")


# Authentication decorator to protect routes
def require_access(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def process_file(file):
    """Extract text from uploaded files (PDF, DOCX, TXT, JSON)."""
    content = file.read()
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "".join(page.extract_text() + "\n" for page in pdf_reader.pages).strip()
    elif filename.endswith(('.docx', '.doc')):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join(paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip())
    elif filename.endswith('.txt'):
        return content.decode('utf-8').strip()
    elif filename.endswith('.json'):
        json_data = json.loads(content.decode('utf-8'))
        return json_data.get('content', str(json_data)) if isinstance(json_data, dict) else str(json_data)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def extract_json_from_response(response):
    """Extract JSON from AI response, handling potential formatting issues."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start, end = response.find('{'), response.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                return json.loads(response[start:end])
            except:
                pass
        return None


def generate_statements(content):
    """Generate 8 reflection statements using Gibbs' Reflective Cycle."""
    prompt = f"""Generate 8 reflection statements based on Gibbs' Reflective Cycle for the lesson content provided. The statements should cover the six stages: Description, Feelings, Evaluation, Analysis, Conclusion, and Action Plan. Each statement should be a positive assertion specific to the content, allowing students to rate their agreement on a scale of 1 (Strongly Disagree) to 5 (Strongly Agree). Ensure the statements are concise and directly tied to the lesson.

    Format the response as JSON:
    {{
        "statements": [
            {{
                "id": 1,
                "stage": "Description",
                "statement": "I can accurately describe the effects of sunlight on different materials."
            }},
            ...
        ]
    }}

    Lesson content: {content}"""

    response = fireworks_client.chat.completions.create(
        model=FIREWORKS_MODEL_ID,
        messages=[
            {"role": "system", "content": "Return only the requested JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    ).choices[0].message.content

    data = extract_json_from_response(response)
    if not data or 'statements' not in data:
        raise ValueError("Failed to generate valid statements")
    return data['statements']


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login with 9-digit access code."""
    error = None
    if request.method == 'POST':
        access_code = request.form['access_code']
        if access_code == ACCESS_CODE:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid access code'
    return render_template('login.html', error=error)


@app.route('/')
@require_access
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/generate-reflection', methods=['POST'])
@require_access
def generate_reflection():
    """Generate reflection statements from uploaded lesson content."""
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        content = process_file(file)
        statements = generate_statements(content)
        return jsonify({'statements': statements})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
@require_access
def chat():
    """Handle chat messages and return AI responses."""
    try:
        data = request.json

        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400

        user_message = data['message']
        reflection_scores = data.get('reflectionScores', {})

        # Create a prompt that includes context about the user's reflection
        prompt = f"""The user is a student who has just completed a reflection exercise with the following scores:
{json.dumps(reflection_scores, indent=2)}

Their message is: {user_message}

Respond as a helpful educational assistant who can discuss their reflections and answer questions
about their learning. Keep responses encouraging, concise, and educational.
"""

        # Generate response using the AI model
        response = fireworks_client.chat.completions.create(
            model=FIREWORKS_MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful educational assistant for students."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400
        ).choices[0].message.content

        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)