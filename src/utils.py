import json
import requests

def preprocess_http_response(response):
    if isinstance(response, requests.Response):
        content = response.text
    elif isinstance(response, bytes):
        content = response.decode('utf-8')
    elif isinstance(response, dict):
        content = json.dumps(response)
    elif isinstance(response, str):
        content = response
    else:
        content = str(response)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"technical_summary": content, "recent_data": "", "report": ""}
    return {
        "technical_summary": str(data.get("technical_summary", "")),
        "recent_data": str(data.get("recent_data", "")),
        "report": str(data.get("report", ""))
    }

def main(arg1: str) -> dict:
    try:
        data = json.loads(arg1)
        result = preprocess_http_response(data)
    except json.JSONDecodeError:
        if arg1.startswith(('http://', 'https://')):
            response = requests.get(arg1)
            result = preprocess_http_response(response)
        else:
            result = {"technical_summary": arg1, "recent_data": "", "report": ""}
    return result