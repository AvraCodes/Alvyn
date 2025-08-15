from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import json
import subprocess
import re
import logging
from typing import List
from dotenv import load_dotenv
import pandas as pd
import glob
import numpy as np
from io import BytesIO
import base64
import requests

load_dotenv()

app = FastAPI(title="Data Analyst Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def detect_available_data_files():
    """Detect all available data files in the directory"""
    data_files = []
    
    # Common data file extensions
    extensions = ['*.csv', '*.json', '*.xlsx', '*.xls', '*.tsv', '*.txt', '*.parquet', '*.png', '*.jpg', '*.jpeg', '*.gif']
    
    for ext in extensions:
        files = glob.glob(ext)
        data_files.extend(files)
    
    return data_files

def analyze_file_structure(filename):
    """Analyze the structure of a data file"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            return {
                'type': 'CSV',
                'columns': list(df.columns),
                'shape': f"{len(df)} rows x {len(df.columns)} columns",
                'sample_data': df.head(3).to_dict('records')
            }
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return {
                        'type': 'JSON Array',
                        'columns': list(data[0].keys()) if isinstance(data[0], dict) else 'N/A',
                        'shape': f"{len(data)} records",
                        'sample_data': data[:3]
                    }
                else:
                    return {
                        'type': 'JSON Object',
                        'structure': str(type(data)),
                        'sample_data': str(data)[:500] if len(str(data)) > 500 else data
                    }
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename)
            return {
                'type': 'Excel',
                'columns': list(df.columns),
                'shape': f"{len(df)} rows x {len(df.columns)} columns",
                'sample_data': df.head(3).to_dict('records')
            }
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return {
                'type': 'Image',
                'filename': filename,
                'description': f"Image file: {filename}"
            }
        elif filename.endswith('.txt'):
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return {
                    'type': 'Text',
                    'size': f"{len(content)} characters",
                    'sample_content': content[:300] if len(content) > 300 else content
                }
        else:
            return {'type': 'Unknown', 'filename': filename}
    except Exception as e:
        return {'type': 'Error', 'error': str(e), 'filename': filename}

def extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response"""
    patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```(.*?)```'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if any(keyword in code for keyword in ['import', 'def', 'print', 'json']):
                return code
    
    return None

def get_aipipe_response(prompt: str) -> str:
    """Get response from AIPipe/OpenRouter with proper error handling"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")  # Read from environment
        api_base = os.getenv("OPENAI_API_BASE", "https://aipipe.org/openrouter/v1")
        
        if not api_key:
            logging.error("OPENAI_API_KEY not found in environment variables")
            return ""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use openai/gpt-4.1-nano as requested
        data = {
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                logging.error(f"No choices in response: {result}")
                return ""
        else:
            logging.error(f"AIPipe API error {response.status_code}: {response.text}")
            return ""
            
    except requests.exceptions.Timeout:
        logging.error("AIPipe API timeout")
        return ""
    except Exception as e:
        logging.error(f"AIPipe API error: {str(e)}")
        return ""

def json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    else:
        return obj

def analyze_data(questions_content: str):
    """Main data analysis function - always generates and executes code"""
    try:
        logging.info("Starting code generation for analysis")
        
        # Detect available files
        available_files = detect_available_data_files()
        
        # Get detailed file analysis for code generation
        file_analysis = {}
        for file in available_files:
            file_analysis[file] = analyze_file_structure(file)

        # Create detailed file descriptions
        file_descriptions = []
        for file, analysis in file_analysis.items():
            desc = f"- {file}: {analysis.get('type', 'Unknown')}"
            if 'shape' in analysis:
                desc += f" ({analysis['shape']})"
            if 'columns' in analysis:
                desc += f", columns: {analysis['columns']}"
            file_descriptions.append(desc)

        # Read system prompt
        try:
            with open("prompt.txt", "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = """You are a data analyst agent. Generate Python code to analyze data and answer questions accurately."""

        # Determine output format from questions
        is_json_object = ("JSON object" in questions_content or 
                         "json object" in questions_content.lower() or
                         "return a JSON object" in questions_content.lower())
        output_format = "JSON object" if is_json_object else "JSON array"

        code_prompt = f"""
{system_prompt}

Available data files:
{chr(10).join(file_descriptions)}

Detailed file analysis:
{json.dumps(file_analysis, indent=2)}

Questions to answer:
{questions_content}

Generate a complete Python script that:
1. Loads ALL available data files automatically using the exact filenames shown above
2. Performs the requested analysis precisely
3. Outputs results as a {output_format} using: print(json.dumps(result))

Requirements:
- Import: pandas, numpy, json, matplotlib, seaborn, scipy, glob, base64, io, requests
- Handle different file formats (CSV, JSON, Excel, Images, Text)
- For plots: return base64 PNG data URI starting with "data:image/png;base64," under 100KB
- Use try/except blocks for error handling
- Make calculations robust and accurate
- Convert numpy types to Python native types before JSON serialization
- For web scraping: use pandas.read_html() for Wikipedia tables or requests + BeautifulSoup
- Always output valid JSON at the end
- Handle missing data gracefully
- Use exact column names and file names as provided

Generate ONLY the Python code:
"""

        code_response = get_aipipe_response(code_prompt)
        
        if not code_response:
            return {"error": "Could not generate analysis code"}

        # Extract code
        code = extract_code_from_response(code_response)
        if not code:
            code = code_response

        # Enhanced code with imports and error handling
        enhanced_code = f"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import base64
from io import BytesIO
import warnings
from collections import defaultdict, deque
import requests
try:
    from bs4 import BeautifulSoup
except ImportError:
    pass
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    \"\"\"Convert numpy types to JSON serializable types\"\"\"
    if isinstance(obj, dict):
        return {{k: convert_numpy_types(v) for k, v in obj.items()}}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    error_result = {{"error": f"Analysis failed: {{str(e)}}"}}
    print(json.dumps(error_result))
"""

        # Execute the code
        with open("analysis_script.py", "w") as f:
            f.write(enhanced_code)

        result = subprocess.run(
            ["python", "analysis_script.py"],
            capture_output=True,
            text=True,
            timeout=180
        )

        output = result.stdout.strip() or result.stderr.strip()

        try:
            parsed_result = json.loads(output)
            # Convert any remaining numpy types
            safe_result = json_serializable(parsed_result)
            return safe_result
        except json.JSONDecodeError:
            return {"error": f"Code execution output: {output[:500]}", "available_files": available_files}

    except subprocess.TimeoutExpired:
        return {"error": "Analysis timed out (3 minutes)"}
    except Exception as e:
        logging.error(f"Code generation error: {str(e)}")
        return {"error": str(e)}

@app.post("/")
async def data_analyst_post_endpoint(request: Request):
    """
    POST endpoint that always generates code for analysis
    """
    try:
        raw_body = await request.body()
        content_type = request.headers.get("content-type", "").lower()
        
        logging.info(f"Content-Type: {content_type}")
        logging.info(f"Raw body length: {len(raw_body)}")
        
        questions_content = ""
        uploaded_files = []
        
        # Handle different content types
        if "application/json" in content_type:
            try:
                body = await request.json()
                if body is not None:
                    questions_content = (body.get("questions", "") or 
                                       body.get("question", "") or 
                                       body.get("query", "") or 
                                       body.get("text", "") or
                                       body.get("prompt", "") or
                                       body.get("input", ""))
                    
                    if not questions_content and isinstance(body, dict):
                        vars_dict = body.get("vars", {})
                        if isinstance(vars_dict, dict):
                            questions_content = str(vars_dict.get("question", ""))
                        
                    if not questions_content:
                        questions_content = str(body)
            except Exception as json_e:
                logging.error(f"JSON parsing error: {json_e}")
                questions_content = raw_body.decode('utf-8', errors='ignore').strip()
                
        elif "multipart/form-data" in content_type:
            try:
                form = await request.form()
                if form is not None:
                    logging.info(f"Form keys: {list(form.keys())}")
                    
                    # Extract questions.txt
                    questions_file = form.get("questions.txt")
                    if questions_file and hasattr(questions_file, 'read'):
                        content = await questions_file.read()
                        questions_content = content.decode('utf-8', errors='ignore').strip()
                        logging.info(f"Found questions.txt file with {len(questions_content)} chars")
                    
                    if not questions_content:
                        questions_content = (form.get("questions") or 
                                           form.get("question") or 
                                           form.get("query") or 
                                           form.get("text") or
                                           form.get("prompt"))
                        if questions_content:
                            questions_content = str(questions_content)
                    
                    # Process ALL additional files (data.csv, image.png, etc.)
                    for key, file in form.items():
                        if key != "questions.txt" and hasattr(file, 'filename') and file.filename:
                            try:
                                content = await file.read()
                                with open(file.filename, "wb") as f:
                                    f.write(content)
                                uploaded_files.append(file.filename)
                                logging.info(f"Saved uploaded file: {file.filename} ({len(content)} bytes)")
                            except Exception as file_e:
                                logging.error(f"Error saving file {file.filename}: {file_e}")
                                
                else:
                    logging.warning("Form data is None")
                    
            except Exception as form_e:
                logging.error(f"Form parsing error: {form_e}")
                questions_content = raw_body.decode('utf-8', errors='ignore').strip()
                
        else:
            try:
                questions_content = raw_body.decode('utf-8', errors='ignore').strip()
            except Exception as decode_e:
                logging.error(f"Decode error: {decode_e}")
                questions_content = str(raw_body)
        
        if questions_content is None:
            questions_content = ""
            
        logging.info(f"Extracted questions: {str(questions_content)[:200]}...")
        logging.info(f"Uploaded files: {uploaded_files}")
        
        if not questions_content or not str(questions_content).strip():
            logging.warning("No questions content found")
            return JSONResponse(
                content={
                    "error": "No questions found in request", 
                    "content_type": content_type,
                    "body_length": len(raw_body),
                    "uploaded_files": uploaded_files
                }, 
                status_code=400
            )

        # Always use code generation approach
        result = analyze_data(str(questions_content))
        
        # Ensure result is JSON serializable before returning
        safe_result = json_serializable(result)
        
        return JSONResponse(content=safe_result)

    except Exception as e:
        logging.error(f"POST endpoint error: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, 
            status_code=500
        )

# Serve the frontend HTML file
@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file"""
    return FileResponse("index.html")
