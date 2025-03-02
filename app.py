from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import docx
import PyPDF2
import io
import functools
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Hardcoded access code (in production, store securely)
ACCESS_CODE = "111111111"

# Initialize Fireworks API client (used for both agents)
fireworks_client = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)

FIREWORKS_MODEL_ID = os.getenv("FIREWORKS_MODEL_ID", "accounts/fireworks/models/llama-v3p1-70b-instruct")

# Gibbs Reflection Framework Definition
GIBBS_FRAMEWORK = {
    "name": "Gibbs' Reflective Cycle",
    "stages": ["Description", "Feelings", "Evaluation", "Analysis", "Conclusion", "Action Plan"],
    "description": "A cyclical model for reflecting on experiences"
}


# Authentication decorator
def require_access(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def process_file(file):
    """Extract text from uploaded files (PDF, DOCX, TXT, JSON)."""
    try:
        if not file:
            raise ValueError("No file provided")

        content = file.read()
        filename = file.filename.lower()
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
        if len(content) > MAX_FILE_SIZE:
            raise ValueError("File size exceeds 5MB limit")

        file.seek(0)

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
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise


def extract_json_from_response(response):
    """Extract JSON from AI response, handling malformed outputs."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        start, end = response.find('{'), response.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                return json.loads(response[start:end])
            except:
                import re
                json_pattern = r'(\{.*\})'
                matches = re.findall(json_pattern, response, re.DOTALL)
                if matches:
                    return json.loads(matches[-1])
        return None


def generate_ai_response(prompt, system_prompt):
    """Generate AI response using Fireworks API."""
    try:
        completion = fireworks_client.chat.completions.create(
            model=FIREWORKS_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating AI response: {str(e)}")
        traceback.print_exc()
        raise


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
            error = 'Invalid access code. Please try again.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    """Log out and clear session."""
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@require_access
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/generate-reflection', methods=['POST'])
@require_access
def generate_reflection():
    """Generate Gibbs-based reflection questions using Question Creator Agent."""
    try:
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No file provided'}), 400

        content = process_file(request.files['file'])

        prompt = f"""You are a Question Creator Agent for Gibbs' Reflective Cycle. Generate 6 reflection questions based on the uploaded lesson content, one for each stage: Description, Feelings, Evaluation, Analysis, Conclusion, Action Plan.

For each question:
- Assign it to a specific Gibbs stage.
- Include a 'context' field with a specific quote or reference from the content.
- Craft a question that is concise, specific, and directly tied to the stage.
- Provide 5 multiple-choice options reflecting different levels of engagement.

Format as JSON:
{{
    "questions": [
        {{
            "id": 1,
            "stage": "Description",
            "context": "Specific quote or reference",
            "question": "Question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
        }},
        ...
    ]
}}

Content: {content}"""

        response = generate_ai_response(prompt, system_prompt="Return only the requested JSON.")
        data = extract_json_from_response(response)

        if not data:
            return jsonify({'error': 'Invalid AI response', 'raw_response': response}), 500

        return jsonify(data)
    except Exception as e:
        print(f"Error in generate_reflection: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/chat_response', methods=['POST'])
@require_access
def chat_response():
    """Handle chat interactions using Chat Agent for deeper reflection."""
    try:
        data = request.json
        history_text = "\n".join(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                                 for msg in data.get('conversationHistory', []))

        prompt = f"""You are a Chat Agent for Gibbs' Reflective Cycle. Provide a detailed, elaborative response based on:
- History: {history_text}
- User Input: {data['userInput']}
- Stage: {data['stage']}
- Context: {data['context']}

Avoid generic responses. Be specific, logical, and ask probing questions to guide the student through the reflection process. Example of a good response:
"Right things is logically right, following a law might be a right thing to do but sometimes laws are not perfect in this case being unlawful both negative and positive outcome"
"I love how you're digging deeper into the complexities of morality and the law. You're absolutely right that just because something is lawful, it doesn't necessarily mean it's morally right. And conversely, just because something is unlawful, it doesn't necessarily mean it's morally wrong. This is a classic critique of deontology, and it's one that many philosophers have grappled with. Kant's ideas about moral obligations and duties are based on the idea that certain actions are inherently right or wrong, regardless of their consequences. But, as you pointed out, laws are created by humans and are therefore imperfect. This raises interesting questions about the nature of morality and the law. If laws are not always perfect, then how can we determine what is morally right or wrong? Should we follow the law simply because it is the law, or should we use our own moral compass to guide our actions? You mentioned that being unlawful can have both negative and positive outcomes. Can you think of an example where breaking the law might lead to a positive outcome, and how would you justify that action morally?"

Provide a response for the current interaction."""

        response = generate_ai_response(prompt, system_prompt="Provide a detailed, elaborative response.")
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat_response: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)