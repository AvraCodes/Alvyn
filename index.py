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
    """Create various sample data files for testing"""
    try:
        # Weather CSV
        weather_data = {
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'temperature': [20 + 5*np.sin(i/10) + np.random.normal(0, 2) for i in range(100)],
            'humidity': [60 + 10*np.sin(i/15) + np.random.normal(0, 5) for i in range(100)],
            'pressure': [1013 + 3*np.sin(i/20) + np.random.normal(0, 2) for i in range(100)],
            'wind_speed': [np.random.uniform(0, 20) for _ in range(100)],
            'city': np.random.choice(['New York', 'London', 'Tokyo', 'Sydney'], 100)
        }
        pd.DataFrame(weather_data).to_csv("sample-weather.csv", index=False)
        
        # Sales CSV
        sales_data = {
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 365),
            'sales': [np.random.uniform(100, 1000) for _ in range(365)],
            'quantity': [np.random.randint(1, 100) for _ in range(365)],
            'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 365)
        }
        pd.DataFrame(sales_data).to_csv("sample-sales.csv", index=False)
        
        # Stock prices CSV
        stock_data = {
            'date': pd.date_range('2023-01-01', periods=252, freq='B'),  # Business days
            'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT', 'AMZN'], 252),
            'open': [100 + np.random.normal(0, 5) for _ in range(252)],
            'high': [105 + np.random.normal(0, 5) for _ in range(252)],
            'low': [95 + np.random.normal(0, 5) for _ in range(252)],
            'close': [100 + np.random.normal(0, 5) for _ in range(252)],
            'volume': [np.random.randint(1000000, 10000000) for _ in range(252)]
        }
        pd.DataFrame(stock_data).to_csv("sample-stocks.csv", index=False)
        
        # Customer data JSON
        customer_data = [
            {
                "id": i,
                "name": f"Customer {i}",
                "age": np.random.randint(18, 80),
                "city": np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston']),
                "purchases": np.random.randint(1, 50),
                "total_spent": round(np.random.uniform(100, 5000), 2),
                "join_date": (pd.Timestamp('2020-01-01') + pd.Timedelta(days=np.random.randint(0, 1000))).strftime('%Y-%m-%d')
            }
            for i in range(500)
        ]
        with open("sample-customers.json", "w") as f:
            json.dump(customer_data, f, indent=2)
            
        logging.info("Created sample data files")
        
    except Exception as e:
        logging.error(f"Error creating sample data: {e}")

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
    """Analyze the structure of a data file"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, nrows=5)
            return {
                'type': 'CSV',
                'columns': list(df.columns),
                'shape': f"{len(pd.read_csv(filename))} rows",
                'sample_data': df.to_dict('records')
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
                        'sample_data': data
                    }
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename, nrows=5)
            return {
                'type': 'Excel',
                'columns': list(df.columns),
                'shape': f"{len(pd.read_excel(filename))} rows",
                'sample_data': df.to_dict('records')
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

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini with proper error handling"""
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        
        response = model.generate_content(prompt)
        
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                text = ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                return text.strip()
        
        return ""
    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        return ""

def analyze_data(questions_content: str, additional_files: List[UploadFile] = None):
    """Main data analysis function that handles any type of data"""
    try:
        import numpy as np
        
        # Create sample data if no data files exist
        available_files = detect_available_data_files()
        if not available_files:
            create_sample_data()
            available_files = detect_available_data_files()

        # Analyze available data files
        file_analysis = {}
        for file in available_files:
            file_analysis[file] = analyze_file_structure(file)

        # Read system prompt
        with open("prompt.txt", "r") as f:
            system_prompt = f.read()

        # Enhanced Phase 1: Try direct answers with data context
        data_context = "\n".join([
            f"Available data file: {file} - {analysis.get('type', 'Unknown')}"
            for file, analysis in file_analysis.items()
        ])

        phase1_prompt = f"""
{system_prompt}

Available data files:
{data_context}

Questions to answer:
{questions_content}

Instructions: Try to answer these questions directly using your knowledge. 
If you can provide ALL answers with confidence, respond with ONLY a clean JSON array/object.
If you need to analyze the actual data files, respond with exactly: "NEED_ANALYSIS"

Respond now:
"""

        direct_response = get_gemini_response(phase1_prompt)
        logging.info(f"Phase 1 response: {direct_response[:200]}...")

        # Check if we got direct answers
        if direct_response and direct_response not in ["NEED_ANALYSIS", "NEED_SCRAPING"]:
            try:
                if (direct_response.startswith('[') or direct_response.startswith('{')):
                    answers = json.loads(direct_response)
                    logging.info("Phase 1 successful - returning direct answers")
                    return answers
            except json.JSONDecodeError:
                logging.info("Phase 1 response not valid JSON, proceeding to Phase 2")

        # Phase 2: Generate comprehensive analysis code
        logging.info("Phase 2: Generating data analysis code")
        
        # Create detailed file descriptions
        file_descriptions = []
        for file, analysis in file_analysis.items():
            desc = f"- {file}: {analysis.get('type', 'Unknown')} with {analysis.get('shape', 'unknown size')}"
            if 'columns' in analysis:
                desc += f", columns: {analysis['columns']}"
            file_descriptions.append(desc)

        is_json_object = "respond with a JSON object" in questions_content.lower()
        output_format = "JSON object" if is_json_object else "JSON array"

        phase2_prompt = f"""
{system_prompt}

Available data files:
{chr(10).join(file_descriptions)}

File analysis details:
{json.dumps(file_analysis, indent=2)}

Questions to answer:
{questions_content}

Generate a complete Python script that:
1. Automatically detects and loads ALL available data files (CSV, JSON, Excel, etc.)
2. Handles different file formats appropriately
3. Analyzes the data to answer ALL questions precisely
4. Outputs results as a {output_format} using: print(json.dumps(answers))

Requirements:
- Import: pandas, numpy, json, matplotlib, seaborn, scipy, glob
- Auto-detect file types and load appropriately:
  * CSV files: pd.read_csv()
  * JSON files: json.load() or pd.read_json()
  * Excel files: pd.read_excel()
- Handle all errors gracefully with try/except
- For plots: return base64 PNG data URI under 100KB
- Clean data properly (handle missing values, data types)
- Make calculations robust and accurate
- Use descriptive variable names

Generate ONLY the Python code:
"""

        code_response = get_gemini_response(phase2_prompt)
        
        if not code_response:
            return {"error": "Could not generate analysis code"}

        # Extract and save code
        code = extract_code_from_response(code_response)
        if not code:
            code = code_response

        # Add imports and error handling wrapper
        enhanced_code = f"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    error_result = {{"error": f"Analysis failed: {{str(e)}}"}}
    print(json.dumps(error_result))
"""

        with open("test_scraper.py", "w") as f:
            f.write(enhanced_code)

        logging.info(f"Generated enhanced analysis code ({len(enhanced_code)} chars)")

        # Execute the generated code
        result = subprocess.run(
            ["python", "test_scraper.py"],
            capture_output=True,
            text=True,
            timeout=180
        )

        output = result.stdout.strip()
        if not output:
            output = result.stderr.strip()

        # Parse results
        try:
            answers = json.loads(output)
            logging.info("Code execution successful")
            return answers
        except json.JSONDecodeError:
            return {"error": f"Execution output: {output[:500]}", "available_files": list(file_analysis.keys())}

    except subprocess.TimeoutExpired:
        return {"error": "Analysis timed out (3 minutes exceeded)"}
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return {"error": str(e)}

@app.post("/")
async def data_analyst_post_endpoint(request: Request):
    """
    POST endpoint that handles various content types and any data format
    """
    try:
        content_type = request.headers.get("content-type", "").lower()
        questions_content = ""
        
        # Handle different content types
        if "application/json" in content_type:
            body = await request.json()
            questions_content = body.get("questions", "") or body.get("question", "") or body.get("query", "")
            
            if not questions_content and isinstance(body, str):
                questions_content = body
            elif not questions_content and isinstance(body, dict):
                for key in ["text", "prompt", "message", "data", "input"]:
                    if key in body and body[key]:
                        questions_content = str(body[key])
                        break
                        
        elif "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            questions_content = form.get("questions") or form.get("question") or form.get("query") or form.get("text")
            
        elif "multipart/form-data" in content_type:
            form = await request.form()
            questions_content = form.get("questions") or form.get("question") or form.get("query") or form.get("text")
            
        else:
            # Plain text or other content types
            raw_body = await request.body()
            questions_content = raw_body.decode('utf-8', errors='ignore').strip()
        
        logging.info(f"Content-Type: {content_type}")
        logging.info(f"Extracted questions: {questions_content[:100]}...")
        
        if not questions_content or not str(questions_content).strip():
            return JSONResponse(
                content={
                    "error": "No questions found in request", 
                    "content_type": content_type,
                    "help": "Send questions as JSON {'questions': '...'} or plain text"
                }, 
                status_code=400
            )

        # Analyze data (will auto-detect and handle any available data)
        result = analyze_data(str(questions_content))
        
        return JSONResponse(content=result)

    except json.JSONDecodeError as e:
        return JSONResponse(content={"error": f"Invalid JSON format: {str(e)}"}, status_code=400)
    except Exception as e:
        logging.error(f"POST endpoint error: {str(e)}")
        return JSONResponse(content={"error": f"Internal error: {str(e)}"}, status_code=500)

@app.post("/api/")
async def data_analyst_endpoint(
    questions_txt: UploadFile = File(alias="questions.txt"),
    additional_files: List[UploadFile] = File(default=[])
):
    """Enhanced endpoint that handles any type of uploaded data files"""
    try:
        # Read questions
        questions_content = (await questions_txt.read()).decode("utf-8")
        if not questions_content.strip():
            return JSONResponse(content={"error": "questions.txt is empty"}, status_code=400)

        # Process and save additional files
        file_info = []
        for file in additional_files:
            if file.filename:
                content = await file.read()
                file_info.append({
                    "filename": file.filename,
                    "size": len(content),
                    "type": file.content_type
                })
                # Save file for analysis
                with open(file.filename, "wb") as f:
                    f.write(content)

        logging.info(f"Processing questions with {len(file_info)} additional files")

        # Analyze data
        result = analyze_data(questions_content, additional_files)
        
        return JSONResponse(content=result)

    except Exception as e:
        logging.error(f"Endpoint error: {str(e)}")
        return JSONResponse(content={"error": f"Internal error: {str(e)}"}, status_code=500)

@app.get("/", response_class=FileResponse)
async def get_frontend():
    """GET endpoint returns the frontend HTML"""
    return FileResponse("index.html")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "data-analyst-agent"}

@app.get("/data")
async def list_available_data():
    """Endpoint to see what data files are available"""
    files = detect_available_data_files()
    analysis = {file: analyze_file_structure(file) for file in files}
    return {"available_files": files, "file_analysis": analysis}
