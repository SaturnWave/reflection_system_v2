# app.py
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from openai import OpenAI
import json
import os
from dotenv import load_dotenv
import docx
import PyPDF2
import io
import boto3
import botocore.session
from aws_requests_auth.aws_auth import AWSRequestsAuth
import requests
import functools
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Access code (in a real app, store this securely)
ACCESS_CODE = "111111111"

# Initialize API clients
fireworks_client = OpenAI(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    base_url="https://api.fireworks.ai/inference/v1"
)

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

FIREWORKS_MODEL_ID = os.getenv("FIREWORKS_MODEL_ID", "accounts/fireworks/models/llama-v3p1-70b-instruct")
BEDROCK_MODEL_ID = os.getenv("AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

# Reflection frameworks definitions
REFLECTION_FRAMEWORKS = {
    "gibbs": {
        "name": "Gibbs' Reflective Cycle",
        "stages": ["Description", "Feelings", "Evaluation", "Analysis", "Conclusion", "Action Plan"],
        "description": "A cyclical model for reflecting on experiences"
    },
    "what_so_what": {
        "name": "What? So What? Now What?",
        "stages": ["What?", "So What?", "Now What?"],
        "description": "A simple framework for progressing from description to analysis to action"
    },
    "deal": {
        "name": "DEAL Model for Critical Reflection",
        "stages": ["Describe", "Examine", "Articulate Learning"],
        "description": "Focuses on connecting experiences to learning objectives"
    },
    "5r": {
        "name": "5R Framework",
        "stages": ["Reporting", "Responding", "Relating", "Reasoning", "Reconstructing"],
        "description": "A comprehensive framework from simple reporting to reconstruction"
    },
    "star": {
        "name": "STAR Method",
        "stages": ["Situation", "Task", "Action", "Result"],
        "description": "Structured approach to reflect on specific experiences"
    }
}

# Decorator to require access code
def require_access(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_bedrock_client():
    """Initialize and return an AWS Bedrock client."""
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    return session.client(service_name='bedrock-runtime')

def process_file(file):
    """Process uploaded file and extract content. Supports PDF, DOCX/DOC, TXT, and JSON."""
    try:
        if not file:
            raise ValueError("No file provided")

        # Save the file content
        content = file.read()
        filename = file.filename.lower()
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB limit
        if len(content) > MAX_FILE_SIZE:
            raise ValueError("File size exceeds limit of 5MB")

        # Reset file pointer to beginning
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
    """Extract JSON from AI response, handling potential formatting issues."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON in the response if it's embedded in text
        start, end = response.find('{'), response.rfind('}') + 1
        if start != -1 and end != 0:
            try:
                return json.loads(response[start:end])
            except:
                # If that fails, try to find a properly formatted JSON object with a more relaxed approach
                import re
                json_pattern = r'(\{.*\})'
                matches = re.findall(json_pattern, response, re.DOTALL)
                if matches:
                    for potential_json in matches:
                        try:
                            return json.loads(potential_json)
                        except:
                            continue
        return None

def generate_ai_response(prompt, model_id, system_prompt="Return only the requested JSON."):
    """Generate response using the selected AI model."""
    print(f"Generating response with model: {model_id}")
    
    if "anthropic" in model_id.lower():
        # Use AWS Bedrock with Claude model
        try:
            print("Using AWS Bedrock Claude model")
            bedrock = get_bedrock_client()
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        except Exception as e:
            print(f"Error with Bedrock: {str(e)}")
            traceback.print_exc()
            raise
    elif "llama-3.3" in model_id.lower() or model_id.lower().startswith("groq/"):
        # Use Groq API
        try:
            print("Using Groq API")
            completion = groq_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error with Groq: {str(e)}")
            traceback.print_exc()
            raise
    else:
        # Use Fireworks API
        try:
            print("Using Fireworks API")
            completion = fireworks_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error with Fireworks: {str(e)}")
            traceback.print_exc()
            raise

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle login with access code."""
    error = None
    if request.method == 'POST':
        if request.form['access_code'] == ACCESS_CODE:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid access code. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """Clear the session and redirect to login."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    """Redirect to login if not authenticated, otherwise render the main page."""
    if not session.get('authenticated'):
        return redirect(url_for('login'))
    return render_template('index.html', frameworks=REFLECTION_FRAMEWORKS)

@app.route('/get-frameworks', methods=['GET'])
@require_access
def get_frameworks():
    """Return available reflection frameworks."""
    return jsonify(REFLECTION_FRAMEWORKS)

@app.route('/recommend-framework', methods=['POST'])
@require_access
def recommend_framework():
    """Analyze content and recommend the most suitable reflection framework."""
    try:
        print("="*50)
        print("RECOMMEND FRAMEWORK ENDPOINT CALLED")
        print("="*50)
        
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No file provided'}), 400

        # Process the file
        try:
            content = process_file(request.files['file'])
            print(f"File processed, content length: {len(content)}")
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
        model_id = request.form.get('model', FIREWORKS_MODEL_ID)
        print(f"Using model: {model_id}")
        
        # Create prompt for framework recommendation
        prompt = f"""Analyze this content and recommend the most suitable reflection framework from the options below.
Consider the nature of the activity, learning objectives, and content structure.

Available frameworks:
1. Gibbs' Reflective Cycle: Description, Feelings, Evaluation, Analysis, Conclusion, Action Plan
   Best for: Reflecting on concrete experiences and emotional responses

2. What? So What? Now What?: What?, So What?, Now What?
   Best for: Simple, progressive reflection moving from facts to implications to action

3. DEAL Model: Describe, Examine, Articulate Learning
   Best for: Connecting experiences to specific learning objectives

4. 5R Framework: Reporting, Responding, Relating, Reasoning, Reconstructing
   Best for: Comprehensive reflection from basic reporting to future application

5. STAR Method: Situation, Task, Action, Result
   Best for: Reflecting on specific tasks or achievements with clear outcomes

Return JSON with this structure:
{{
    "recommended_framework": "framework_id",
    "framework_name": "Full Name of Framework",
    "explanation": "Detailed explanation of why this framework is most suitable",
    "content_summary": "Brief summary of the analyzed content"
}}

Where framework_id must be one of: "gibbs", "what_so_what", "deal", "5r", or "star".

Content to analyze: {content}"""

        # Generate AI response
        try:
            print("Generating AI recommendation")
            response = generate_ai_response(
                prompt,
                model_id,
                system_prompt="Analyze content and recommend the most suitable reflection framework."
            )
            print("AI recommendation received")
        except Exception as e:
            print(f"Error generating AI recommendation: {str(e)}")
            return jsonify({'error': f'Error with AI model: {str(e)}'}), 500
        
        # Extract recommendation
        try:
            recommendation = extract_json_from_response(response)
            if not recommendation:
                print("Failed to extract valid JSON from AI response")
                return jsonify({'error': 'Invalid AI response', 'raw_response': response}), 500
            print(f"Recommendation: {recommendation}")
        except Exception as e:
            print(f"Error parsing AI response: {str(e)}")
            return jsonify({'error': f'Error parsing response: {str(e)}', 'raw_response': response}), 500
            
        return jsonify(recommendation)
    except Exception as e:
        print(f"Unexpected error in recommend_framework: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate-reflection', methods=['POST'])
@require_access
def generate_reflection():
    """Generate reflection questions from uploaded lesson content using selected framework."""
    try:
        # Print detailed debug information
        print("="*50)
        print("GENERATE REFLECTION ENDPOINT CALLED")
        print("="*50)
        
        # Check if request has files
        print(f"Request has files: {request.files}")
        print(f"Request form data: {request.form}")
        
        if 'file' not in request.files:
            print("ERROR: No file part in request")
            return jsonify({'error': 'No file part in request'}), 400
            
        file = request.files['file']
        if not file.filename:
            print("ERROR: No file selected")
            return jsonify({'error': 'No file selected'}), 400

        print(f"File received: {file.filename}")
        
        # Process the file and extract content
        try:
            print(f"Attempting to process file {file.filename}")
            content = process_file(file)
            print(f"File processed successfully, content length: {len(content)}")
            print(f"First 100 chars of content: {content[:100]}")
        except Exception as e:
            print(f"ERROR processing file: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        
        framework_id = request.form.get('framework', 'gibbs')
        model_id = request.form.get('model', FIREWORKS_MODEL_ID)
        
        print(f"Framework: {framework_id}, Model: {model_id}")
        
        # Get the selected framework
        framework = REFLECTION_FRAMEWORKS.get(framework_id, REFLECTION_FRAMEWORKS['gibbs'])
        print(f"Using framework: {framework['name']}")
        
        # Create prompt based on the selected framework
        prompt = generate_framework_prompt(framework, content)
        print(f"Prompt generated, length: {len(prompt)}")
        
        # Generate response using selected model
        try:
            print(f"Attempting to generate AI response using model: {model_id}")
            response = generate_ai_response(prompt, model_id)
            print(f"AI response received, length: {len(response)}")
            print(f"First 100 chars of response: {response[:100]}")
        except Exception as e:
            print(f"ERROR generating AI response: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Error with AI model: {str(e)}'}), 500
        
        # Parse the response
        try:
            print("Trying to extract JSON from response")
            data = extract_json_from_response(response)
            if not data:
                print("ERROR: Failed to extract valid JSON from AI response")
                print(f"Raw response: {response}")
                return jsonify({'error': 'Invalid AI response', 'raw_response': response}), 500
            print(f"JSON successfully extracted: {list(data.keys())}")
            if 'questions' in data:
                print(f"Number of questions: {len(data['questions'])}")
        except Exception as e:
            print(f"ERROR parsing AI response: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Error parsing response: {str(e)}', 'raw_response': response}), 500
        
        print("Returning successful response")
        print("="*50)
        return jsonify(data)
        
    except Exception as e:
        print(f"UNEXPECTED ERROR in generate_reflection: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_framework_prompt(framework, content):
    """Generate prompt based on the selected reflection framework."""
    stages_text = ", ".join(framework["stages"])
    
    prompt = f"""Generate 8 reflection questions based on the {framework['name']} ({stages_text}) from this lesson content.

For each question:
- Assign it to one of the stages, ensuring all stages are covered (distribute questions evenly across stages).
- Include a 'context' field with a specific quote or reference from the lesson.
- Craft a question incorporating the context.
- Provide 5 multiple-choice options reflecting different levels of understanding or engagement.

Format as JSON:
{{
    "framework": "{framework['name']}",
    "questions": [
        {{
            "id": 1,
            "stage": "[Stage Name]",
            "context": "Specific quote or reference",
            "question": "Question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
        }},
        ...
    ]
}}

Content: {content}"""
    return prompt

@app.route('/select_option', methods=['POST'])
@require_access
def handle_option_selection():
    """Generate initial feedback based on the selected option."""
    try:
        data = request.json
        model_id = data.get('model', FIREWORKS_MODEL_ID)
        
        prompt = f"""Provide encouraging feedback based on:
- Selected option: {data['selectedOption']}
- Stage: {data['stage']}
- Context: {data['context']}

Return JSON:
{{
    "acknowledgment": "Personalized acknowledgment",
    "encouragement": "Specific encouragement",
    "deeperThought": "Thought-provoking question"
}}"""

        response = generate_ai_response(
            prompt,
            model_id,
            system_prompt="Return encouraging JSON feedback."
        )
        
        feedback = extract_json_from_response(response)
        return jsonify(feedback) if feedback else jsonify({'error': 'Invalid feedback', 'raw_response': response}), 500
    except Exception as e:
        print(f"Error in handle_option_selection: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/chat_response', methods=['POST'])
@require_access
def chat_response():
    """Handle chat interactions for deeper reflection."""
    try:
        data = request.json
        model_id = data.get('model', FIREWORKS_MODEL_ID)
        
        history_text = "\n".join(f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                                for msg in data.get('conversationHistory', []))
        
        prompt = f"""Respond as a supportive guide based on:
- History: {history_text}
- User Input: {data['userInput']}
- Stage: {data['stage']}
- Context: {data['context']}

Be conversational, tie to the context, and end with a question."""

        response = generate_ai_response(
            prompt,
            model_id,
            system_prompt="Provide educational, supportive responses."
        )
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat_response: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)