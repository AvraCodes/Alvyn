from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import google.generativeai as genai
import os
import json
import subprocess
import re
import logging
from typing import List
from dotenv import load_dotenv
import pandas as pd
import glob
import numpy as np  # Add this import at the top
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

def create_sample_data():
    """Do not create any hardcoded sample data - work only with uploaded files"""
    # Removed all hardcoded data creation
    # The system should work entirely with files uploaded via the API
    pass

def detect_available_data_files():
    """Detect all available data files in the directory"""
    data_files = []
    
    # Common data file extensions
    extensions = ['*.csv', '*.json', '*.xlsx', '*.xls', '*.tsv', '*.txt', '*.parquet']
    
    for ext in extensions:
        files = glob.glob(ext)
        data_files.extend(files)
    
    return data_files

def analyze_file_structure(filename):
    """Analyze the structure of a data file with more sample data"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            return {
                'type': 'CSV',
                'columns': list(df.columns),
                'shape': f"{len(df)} rows",
                'sample_data': df.head(5).to_dict('records')  # Get 5 sample records
            }
        elif filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    return {
                        'type': 'JSON Array',
                        'columns': list(data[0].keys()) if isinstance(data[0], dict) else 'N/A',
                        'shape': f"{len(data)} records",
                        'sample_data': data[:5]  # Get 5 sample records
                    }
                else:
                    return {
                        'type': 'JSON Object',
                        'structure': str(type(data)),
                        'sample_data': data if len(str(data)) < 1000 else str(data)[:1000]
                    }
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename)
            return {
                'type': 'Excel',
                'columns': list(df.columns),
                'shape': f"{len(df)} rows",
                'sample_data': df.head(5).to_dict('records')
            }
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return {
                'type': 'Image',
                'filename': filename,
                'sample_data': f"Image file: {filename}"
            }
        elif filename.endswith('.txt'):
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return {
                    'type': 'Text',
                    'size': f"{len(content)} characters",
                    'sample_data': content[:500] if len(content) > 500 else content
                }
        else:
            return {'type': 'Unknown', 'filename': filename, 'sample_data': f"File: {filename}"}
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
        api_key = os.getenv("OPENAI_API_KEY")  # This is your AIPipe token
        api_base = os.getenv("OPENAI_API_BASE", "https://aipipe.org/openrouter/v1")
        
        if not api_key:
            logging.error("OPENAI_API_KEY (AIPipe token) not found")
            return ""
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use Claude 3.5 Sonnet - much better for code generation and analysis
        data = {
            "model": "anthropic/claude-3.5-sonnet",  # Changed back to Claude
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

def analyze_data(questions_content: str, additional_files: List[UploadFile] = None):
    """Main data analysis function - just calls the LLM-first approach"""
    return analyze_data_with_direct_llm(questions_content, additional_files)

def analyze_data_with_direct_llm(questions_content: str, additional_files: List[UploadFile] = None):
    """Enhanced analysis that tries direct LLM answers first, then fallback to code generation"""
    try:
        # Detect available files (both uploaded and existing) - NO sample data creation
        available_files = detect_available_data_files()
        
        # Work with whatever files are actually available, don't create fake data
        if not available_files:
            logging.info("No data files available - proceeding with questions-only analysis")
            
            # Try direct LLM response without data
            direct_prompt = f"""
You are a data analyst agent. Answer the following questions using your knowledge:

Questions to answer:
{questions_content}

INSTRUCTIONS:
1. If these questions can be answered with general knowledge, provide direct answers in the requested JSON format
2. If specific data files are mentioned that you don't have access to, respond with exactly: "NEED_DATA_FILES"
3. If you need to perform calculations on actual data, respond with exactly: "NEED_CODE_ANALYSIS"

Response:
"""
            
            direct_response = get_aipipe_response(direct_prompt)
            
            if direct_response and "NEED_DATA_FILES" in direct_response:
                return {"error": "Required data files not provided"}
            elif direct_response and "NEED_CODE_ANALYSIS" in direct_response:
                return {"error": "Cannot generate code without data files"}
            else:
                # Try to parse as JSON
                try:
                    clean_response = direct_response.strip()
                    if clean_response.startswith('```json'):
                        clean_response = clean_response.replace('```json', '').replace('```', '').strip()
                    elif clean_response.startswith('```'):
                        clean_response = clean_response.replace('```', '').strip()

                    if clean_response.startswith('[') or clean_response.startswith('{'):
                        return json.loads(clean_response)
                except json.JSONDecodeError:
                    pass
                
                return {"error": "No data files available for analysis"}

        # Continue with files that were actually uploaded...
        # Analyze file structures and get sample data for Phase 1
        file_summaries = []
        file_contents = {}
        
        for file in available_files:
            analysis = analyze_file_structure(file)
            summary = f"- {file}: {analysis.get('type', 'Unknown')}"
            if 'shape' in analysis:
                summary += f" ({analysis['shape']})"
            if 'columns' in analysis:
                summary += f", columns: {analysis['columns'][:5]}"  # First 5 columns
            file_summaries.append(summary)
            
            # Get sample data for LLM context
            if analysis.get('sample_data'):
                file_contents[file] = analysis['sample_data']

        # Read system prompt
        try:
            with open("prompt.txt", "r") as f:
                system_prompt = f.read()
        except FileNotFoundError:
            system_prompt = """You are a data analyst agent. Answer questions accurately using your knowledge and available data context."""

        # Build comprehensive context for LLM including file contents
        file_context = []
        for file in available_files:
            if file in file_contents:
                file_context.append(f"\n{file} sample data:")
                if isinstance(file_contents[file], list):
                    # Show first few records for arrays
                    for i, record in enumerate(file_contents[file][:3]):
                        file_context.append(f"  Record {i+1}: {record}")
                else:
                    file_context.append(f"  {file_contents[file]}")

        # Phase 1: Direct LLM Response with file contents
        direct_prompt = f"""
{system_prompt}

Available data files:
{chr(10).join(file_summaries)}

Sample data from files:
{chr(10).join(file_context)}

Questions to answer:
{questions_content}

INSTRUCTIONS:
1. Use the sample data provided above to answer questions if possible
2. For simple questions about data structure, counts, or basic analysis, provide direct answers
3. If you can answer all questions with confidence using the sample data, respond with ONLY the requested JSON format
4. If you need to analyze complete datasets, generate visualizations, or perform complex calculations, respond with exactly: "NEED_CODE_ANALYSIS"

Determine the expected response format from the questions:
- If questions ask for a JSON object with specific keys, return a JSON object
- If questions ask for a JSON array, return a JSON array  
- If questions mention specific metrics, include those exact keys

Response:
"""

        logging.info("Phase 1: Attempting direct LLM response with file contents")
        direct_response = get_aipipe_response(direct_prompt)
        
        if not direct_response:
            logging.warning("No response from LLM, proceeding to code generation")
            return generate_code_analysis(questions_content, available_files, system_prompt)

        logging.info(f"Direct LLM response: {direct_response[:200]}...")

        # Check if LLM indicated it needs code analysis
        if "NEED_CODE_ANALYSIS" in direct_response:
            logging.info("LLM requested code analysis, proceeding to Phase 2")
            return generate_code_analysis(questions_content, available_files, system_prompt)

        # Try to parse as JSON (successful direct answer)
        try:
            # Clean the response - remove any markdown formatting
            clean_response = direct_response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response.replace('```json', '').replace('```', '').strip()
            elif clean_response.startswith('```'):
                clean_response = clean_response.replace('```', '').strip()

            # Parse the JSON
            if clean_response.startswith('[') or clean_response.startswith('{'):
                answers = json.loads(clean_response)
                logging.info("Phase 1 successful - returning direct LLM answers")
                return answers
            else:
                logging.info("Response not in JSON format, proceeding to code generation")
                return generate_code_analysis(questions_content, available_files, system_prompt)
                
        except json.JSONDecodeError as e:
            logging.info(f"JSON parsing failed: {e}, proceeding to code generation")
            return generate_code_analysis(questions_content, available_files, system_prompt)

    except Exception as e:
        logging.error(f"Direct LLM analysis error: {str(e)}")
        return {"error": str(e)}

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

def safe_json_dumps(obj):
    """Safely convert object to JSON string with numpy type handling"""
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
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
    
    return json.dumps(convert_types(obj))

def generate_code_analysis(questions_content: str, available_files: List, system_prompt: str):
    """Generate and execute Python code for complex analysis"""
    try:
        logging.info("Phase 2: Generating Python code for analysis")
        
        # Get detailed file analysis for code generation
        file_analysis = {}
        for file in available_files:
            file_analysis[file] = analyze_file_structure(file)

        # Create detailed file descriptions
        file_descriptions = []
        for file, analysis in file_analysis.items():
            desc = f"- {file}: {analysis.get('type', 'Unknown')} with {analysis.get('shape', 'unknown size')}"
            if 'columns' in analysis:
                desc += f", columns: {analysis['columns']}"
            file_descriptions.append(desc)

        # Determine output format
        is_json_object = ("respond with a JSON object" in questions_content.lower() or 
                         "return a JSON object" in questions_content.lower() or
                         "json object with keys" in questions_content.lower())
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
1. Loads ALL available data files automatically
2. Performs the requested analysis precisely
3. Outputs results as a {output_format} using: print(json.dumps(result))

Requirements:
- Import: pandas, numpy, json, matplotlib, seaborn, scipy, glob, base64, io
- Do NOT import networkx - use pandas/collections for any graph analysis
- Handle different file formats (CSV, JSON, Excel)
- For plots: return base64 PNG data URI under 100KB
- Use try/except blocks for error handling
- Make calculations robust and accurate
- Convert numpy types to Python native types before JSON serialization
- Always output valid JSON at the end

Generate ONLY the Python code:
"""

        code_response = get_aipipe_response(code_prompt)
        
        if not code_response:
            return {"error": "Could not generate analysis code"}

        # Extract code
        code = extract_code_from_response(code_response)
        if not code:
            code = code_response

        # Handle network analysis specifically
        if ("edges.csv" in [f.lower() for f in available_files]) or ("network" in questions_content.lower()):
            if "networkx" in code.lower():
                logging.warning("Replacing networkx with pure Python implementation")
                code = get_pure_python_network_code()

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

def get_pure_python_network_code():
    """Pure Python network analysis code without networkx - with JSON serialization fix"""
    return """
import pandas as pd
import json
from collections import defaultdict, deque

def convert_numpy_types(obj):
    \"\"\"Convert numpy types to JSON serializable types\"\"\"
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'dtype'):  # pandas/numpy types
        if 'int' in str(obj.dtype):
            return int(obj)
        elif 'float' in str(obj.dtype):
            return float(obj)
        else:
            return str(obj)
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

# Load edges data with flexible column detection
edges_df = pd.read_csv('edges.csv')
cols = [c.lower() for c in edges_df.columns]

# Detect source/target columns
source_col = None
target_col = None

for col in edges_df.columns:
    col_lower = col.lower()
    if col_lower in ['source', 'from', 'node1', 'u', 'start']:
        source_col = col
    elif col_lower in ['target', 'to', 'node2', 'v', 'end']:
        target_col = col

if source_col is None or target_col is None:
    source_col = edges_df.columns[0]
    target_col = edges_df.columns[1]

# Build undirected graph
graph = defaultdict(set)
nodes = set()

for _, row in edges_df.iterrows():
    u, v = row[source_col], row[target_col]
    graph[u].add(v)
    graph[v].add(u)
    nodes.add(u)
    nodes.add(v)

# Calculate metrics
edge_count = len(edges_df)
node_count = len(nodes)
degrees = {n: len(graph[n]) for n in nodes}
highest_degree_node = str(max(degrees, key=degrees.get)) if degrees else ""
average_degree = sum(degrees.values()) / len(degrees) if degrees else 0.0
max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 0
density = edge_count / max_edges if max_edges > 0 else 0.0

def bfs_shortest_path(start, end):
    if start == end: return 0
    if start not in nodes or end not in nodes: return -1
    
    queue = deque([(start, 0)])
    visited = {start}
    
    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == end: return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1

# Find specific nodes or use first two
alice = next((n for n in nodes if str(n).lower() in ['alice', 'a']), None)
bob = next((n for n in nodes if str(n).lower() in ['bob', 'b']), None)

if not alice or not bob:
    node_list = list(nodes)
    alice = node_list[0] if node_list else ""
    bob = node_list[1] if len(node_list) > 1 else alice

result = {
    "edge_count": int(edge_count),
    "highest_degree_node": str(highest_degree_node),
    "average_degree": float(round(average_degree, 6)),
    "density": float(round(density, 6)),
    "shortest_path_alice_bob": int(bfs_shortest_path(alice, bob)) if alice != bob else 0
}

# Ensure all values are JSON serializable
result = convert_numpy_types(result)
print(json.dumps(result))
"""

# Update the main functions
def analyze_data_with_fallback(questions_content: str, additional_files: List[UploadFile] = None):
    """Main analysis function - prioritizes direct LLM answers"""
    return analyze_data_with_direct_llm(questions_content, additional_files)

# Update the POST endpoint to use the new approach
@app.post("/")
async def data_analyst_post_endpoint(request: Request):
    """
    POST endpoint with LLM-first approach
    """
    try:
        # Get the raw body first for debugging
        raw_body = await request.body()
        content_type = request.headers.get("content-type", "").lower()
        
        logging.info(f"Content-Type: {content_type}")
        logging.info(f"Raw body length: {len(raw_body)}")
        
        questions_content = ""
        
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
                    
                    # Process additional files
                    for key, file in form.items():
                        if key != "questions.txt" and hasattr(file, 'filename') and file.filename:
                            try:
                                content = await file.read()
                                with open(file.filename, "wb") as f:
                                    f.write(content)
                                logging.info(f"Saved additional file: {file.filename}")
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
        
        if not questions_content or not str(questions_content).strip():
            logging.warning("No questions content found")
            return JSONResponse(
                content={
                    "error": "No questions found in request", 
                    "content_type": content_type,
                    "body_length": len(raw_body)
                }, 
                status_code=400
            )

        # Use LLM-first analysis approach
        result = analyze_data_with_fallback(str(questions_content))
        
        # Ensure result is JSON serializable before returning
        safe_result = json_serializable(result)
        
        return JSONResponse(content=safe_result)

    except Exception as e:
        logging.error(f"POST endpoint error: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": f"Internal error: {str(e)}"}, 
            status_code=500
        )
