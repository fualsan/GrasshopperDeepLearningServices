from openai import OpenAI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import datetime
import os
import uvicorn
from time import time_ns

from dotenv import load_dotenv
load_dotenv()


client = OpenAI(
    api_key=os.getenv('OPENAI_KEY')
)

################# CONVERSATION ###################
INIT_CONTENT = {
    'role': 'developer',
    'content': 'You are a software engineer working on computational design. You will write Python code that runs on Rhino Grasshopper. Respond only the generated Python code, do not respond anything else. Code should be in ```python ... ``` brackets.'
}


conversation = [INIT_CONTENT]


def reset_conversation_history():
    global conversation
    conversation = [INIT_CONTENT]
##################################################


# uvicorn runs this
app = FastAPI()


class ScriptGenerationRequest(BaseModel):
	prompt: str
	model: str = Field(default='gpt-4o')
	temperature: float = Field(default=0.8, gt=0, description='Temperature must be between 0.0 and 2.0')


@app.post('/generate_script')
async def generate_script_endpoint(request: ScriptGenerationRequest):
    global conversation
    
    conversation.append({'role': 'user', 'content': request.prompt})
    
    # response in: chat_completion.choices[0].message.content
    # response role (assistant) in: chat_completion.choices[0].message.role 
    chat_completion = client.chat.completions.create(
        model=request.model,
        messages=conversation,
        temperature=request.temperature,
    )

    response_content = chat_completion.choices[0].message.content
    conversation.append({'role': 'assistant', 'content': response_content})

    try:
        script_from_content = response_content.split('```python')[1].split('```')[0]
    except:
        script_from_content = ''

    if len(script_from_content) < 1:
        eneration_status = 'error'
    else:
        generation_status = 'ok'
    
    response = {
        'generation_status': generation_status,
        'generated_script': script_from_content,
    }
    
    return response


# NOTE: GET REQUEST, NOT POST!
@app.get('/get_conversation_history')
async def get_conversation_history_endpoint():
    global conversation
    
    response = {
        'conversation_history': conversation,
    }
    
    return response


# NOTE: GET REQUEST, NOT POST!
@app.get('/reset_conversation_history')
async def reset_conversation_history_endpoint():

    try:
        reset_conversation_history()
        reset_conversation_status = 'ok'
    except:
        reset_conversation_status = 'error'
    
    response = {
        'reset_conversation_status': reset_conversation_status
    }
    
    return response


@app.middleware('http')
async def log_request_info(request, call_next):
	"""
	request.client.host
    request.client.port
    request.method
    request.url
    request.url.path
    request.query_params
    request.headers
	"""
	print(f'{request.url.path} endpoint received {request.method} request from {request.client.host}:{request.client.port} using agent: {request.headers["user-agent"]}')

	response = await call_next(request)
	return response


if __name__ == '__main__':
    print(f'GH Scripting Agent Grasshopper Deep Learning Services (GHDLS) is starting...')
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=10000)