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
import requests
import functools
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Use environment variable for secret key (important for Vercel)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-replace-in-production')

# Access code from environment variables
ACCESS_CODE = os.environ.get('ACCESS_CODE', '111111111')

# Initialize API clients with better error handling
try:
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")
    if fireworks_api_key:
        fireworks_client = OpenAI(
            api_key=fireworks_api_key,
            base_url="https://api.fireworks.ai/inference/v1"
        )
    else:
        fireworks_client = None
        logger.warning("FIREWORKS_API_KEY not set")

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if groq_api_key:
        groq_client = OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        groq_client = None
        logger.warning("GROQ_API_KEY not set")
except Exception as e:
    logger.error(f"Error initializing API clients: {str(e)}")
    fireworks_client = None
    groq_client = None

FIREWORKS_MODEL_ID = os.environ.get("FIREWORKS_MODEL_ID", "accounts/fireworks/models/llama-v3p1-70b-instruct")
BEDROCK_MODEL_ID = os.environ.get("AWS_BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
GROQ_MODEL_ID = os.environ.get("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

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

# Decorator to require access code - with better error handling for Vercel
def require_access(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if not session.get('authenticated'):
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Session error in require_access: {str(e)}")
            # Fallback for serverless environment if session fails
            return redirect(url_for('login'))
    return decorated_function

def get_bedrock_client():
    """Initialize and return an AWS Bedrock client with better error handling."""
    try:
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")
        
        if not aws_access_key or not aws_secret_key:
            logger.warning("AWS credentials not set")
            return None
            
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        return session.client(service_name='bedrock-runtime')
    except Exception as e:
        logger.error(f"Error creating Bedrock client: {str(e)}")
        return None

def process_file(file):
    """Process uploaded file with improved error handling for Vercel."""
    try:
        if not file:
            raise ValueError("No file provided")

        # Check if file is accessible
        if not hasattr(file, 'read'):
            raise ValueError("File object is not readable")

        # Save the file content
        content = file.read()
        
        if not content:
            raise ValueError("File is empty")
            
        filename = file.filename.lower() if hasattr(file, 'filename') else "unknown.txt"
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
        logger.error(f"Error processing file: {str(e)}")
        traceback.print_exc()
        raise

def extract_json_from_response(response):
    """Extract JSON from AI response with better error handling."""
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
    """Generate response using the selected AI model with improved error handling."""
    logger.info(f"Generating response with model: {model_id}")
    
    if "anthropic" in model_id.lower():
        # Use AWS Bedrock with Claude model
        try:
            bedrock = get_bedrock_client()
            if not bedrock:
                return json.dumps({
                    "error": "AWS Bedrock client not available"
                })
                
            logger.info("Using AWS Bedrock Claude model")
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
            logger.error(f"Error with Bedrock: {str(e)}")
            traceback.print_exc()
            return json.dumps({
                "error": f"AWS Bedrock error: {str(e)}"
            })
    elif "llama-3.3" in model_id.lower() or model_id.lower().startswith("groq/"):
        # Use Groq API
        try:
            if not groq_client:
                return json.dumps({
                    "error": "Groq client not available"
                })
                
            logger.info("Using Groq API")
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
            logger.error(f"Error with Groq: {str(e)}")
            traceback.print_exc()
            return json.dumps({
                "error": f"Groq API error: {str(e)}"
            })
    else:
        # Use Fireworks API
        try:
            if not fireworks_client:
                return json.dumps({
                    "error": "Fireworks client not available"
                })
                
            logger.info("Using Fireworks API")
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
            logger.error(f"Error with Fireworks: {str(e)}")
            traceback.print_exc()
            return json.dumps({
                "error": f"Fireworks API error: {str(e)}"
            })

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
        logger.info("RECOMMEND FRAMEWORK ENDPOINT CALLED")
        
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No file provided'}), 400

        # Process the file
        try:
            content = process_file(request.files['file'])
            logger.info(f"File processed, content length: {len(content)}")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
            
        model_id = request.form.get('model', FIREWORKS_MODEL_ID)
        logger.info(f"Using model: {model_id}")
        
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
            logger.info("Generating AI recommendation")
            response = generate_ai_response(
                prompt,
                model_id,
                system_prompt="Analyze content and recommend the most suitable reflection framework."
            )
            logger.info("AI recommendation received")
        except Exception as e:
            logger.error(f"Error generating AI recommendation: {str(e)}")
            return jsonify({'error': f'Error with AI model: {str(e)}'}), 500
        
        # Extract recommendation
        try:
            recommendation = extract_json_from_response(response)
            if not recommendation:
                logger.error("Failed to extract valid JSON from AI response")
                return jsonify({'error': 'Invalid AI response', 'raw_response': response}), 500
            logger.info(f"Recommendation: {recommendation}")
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return jsonify({'error': f'Error parsing response: {str(e)}', 'raw_response': response}), 500
            
        return jsonify(recommendation)
    except Exception as e:
        logger.error(f"Unexpected error in recommend_framework: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate-reflection', methods=['POST'])
@require_access
def generate_reflection():
    """Generate reflection questions from uploaded lesson content using selected framework."""
    try:
        logger.info("GENERATE REFLECTION ENDPOINT CALLED")
        
        # Check if request has files
        logger.info(f"Request has files: {request.files}")
        logger.info(f"Request form data: {request.form}")
        
        if 'file' not in request.files:
            logger.error("ERROR: No file part in request")
            return jsonify({'error': 'No file part in request'}), 400
            
        file = request.files['file']
        if not file.filename:
            logger.error("ERROR: No file selected")
            return jsonify({'error': 'No file selected'}), 400

        logger.info(f"File received: {file.filename}")
        
        # Process the file and extract content
        try:
            logger.info(f"Attempting to process file {file.filename}")
            content = process_file(file)
            logger.info(f"File processed successfully, content length: {len(content)}")
        except Exception as e:
            logger.error(f"ERROR processing file: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        
        framework_id = request.form.get('framework', 'gibbs')
        model_id = request.form.get('model', FIREWORKS_MODEL_ID)
        
        logger.info(f"Framework: {framework_id}, Model: {model_id}")
        
        # Get the selected framework
        framework = REFLECTION_FRAMEWORKS.get(framework_id, REFLECTION_FRAMEWORKS['gibbs'])
        logger.info(f"Using framework: {framework['name']}")
        
        # Create prompt based on the selected framework
        prompt = generate_framework_prompt(framework, content)
        logger.info(f"Prompt generated, length: {len(prompt)}")
        
        # Generate response using selected model
        try:
            logger.info(f"Attempting to generate AI response using model: {model_id}")
            response = generate_ai_response(prompt, model_id)
            logger.info(f"AI response received, length: {len(response)}")
        except Exception as e:
            logger.error(f"ERROR generating AI response: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Error with AI model: {str(e)}'}), 500
        
        # Parse the response
        try:
            logger.info("Trying to extract JSON from response")
            data = extract_json_from_response(response)
            if not data:
                logger.error("ERROR: Failed to extract valid JSON from AI response")
                return jsonify({'error': 'Invalid AI response', 'raw_response': response}), 500
            logger.info(f"JSON successfully extracted: {list(data.keys())}")
            if 'questions' in data:
                logger.info(f"Number of questions: {len(data['questions'])}")
        except Exception as e:
            logger.error(f"ERROR parsing AI response: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Error parsing response: {str(e)}', 'raw_response': response}), 500
        
        logger.info("Returning successful response")
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR in generate_reflection: {str(e)}")
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
        logger.error(f"Error in handle_option_selection: {str(e)}")
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
        logger.error(f"Error in chat_response: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Vercel-specific middleware for proper cache control
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

# Note: Don't include the app.run() section for Vercel deployment
# if __name__ == '__main__':
#     app.run(debug=True)