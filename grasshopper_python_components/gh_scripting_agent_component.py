import base64
import json
from urllib import request


# LOCAL URLS
GENERATE_SCRIPT_URL = 'http://127.0.0.1:10000/generate_script'
GET_CONVERSATION_HISTORY_URL = 'http://127.0.0.1:10000/get_conversation_history'
RESET_CONVERSATION_HISTORY_URL = 'http://127.0.0.1:10000/reset_conversation_history'


def send_request_generate_script():
    
    # Prepare the data to be sent
    request_json = {
        'prompt': prompt,
        'model': model,
        'temperature': temperature
    }
    
    response = requests.post(GENERATE_SCRIPT_URL, json=request_json)
    
    if response.status_code == 200:
        response_json = response.json()

        if response_json['generation_status'] == 'ok':
            generated_script = response_json['generated_script']
            print('Generated Script:')
            print(generated_script)
            
            try:
                print('*** RUNNING GENERATED SCRIPT ***')
                exec(generated_script)
                print('*** RUNNING ENDED ***')
            except Exception as e:
                print(f'Error executing the script: {e}')
            
        else:
            print('Error in generating the code, please check the prompt!')
            print(f'SENT PROMPT: {prompt}')
    else:
        print(f'Error: {response.status_code}, {response.text}')


def send_request_get_conversation_history():
    response_json = requests.get(GET_CONVERSATION_HISTORY_URL).json()
    conversation_history = response_json['conversation_history']
    for c in conversation_history:
        print(c['role'].upper())
        print(c['content'])
        print()


def send_request_reset_conversation_history():
    response_json = requests.get(RESET_CONVERSATION_HISTORY_URL).json()
    reset_conversation_status = response_json['reset_conversation_status']
    print(f'reset_conversation_status: {reset_conversation_status}')


if generate_script:
    send_request()

if get_conversation_history:
    send_request_get_conversation_history()

if reset_conversation_history:
    send_request_reset_conversation_history()

