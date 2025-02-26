from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
import json
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import the Flask app from app.py
    from app import app as flask_app
except Exception as e:
    # Create a fallback app if import fails
    flask_app = Flask(__name__)
    
    @flask_app.route('/')
    def error_page():
        return jsonify({
            "error": f"Failed to import app: {str(e)}",
            "hint": "Check your environment variables and dependencies."
        }), 500

# This is required for Vercel serverless functions
app = flask_app