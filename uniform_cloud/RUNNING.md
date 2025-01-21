# Installing and Running Uniform Cloud Server

## Step 1: Create and Activate Virtual Environment

### Initialize new virtual environment
```bash
python -m venv venv
```

### Activate virtual environment
```bash
source venv/bin/activate
```

## Step 3: Install Required Libraries
```bash
pip install requests openai fastapi pydantic pillow uvicorn python-dotenv
```

## Step 4: Setup API Keys
Create a .env file in the root of your project to store sensitive data like API keys. Here's an example .env file:

```bash
STABILITY_API_KEY=your_stability_ai_key
OPENAI_API_KEY=your_openai_api_key
```

## Step 5: Run Server
The command below will prevent the server stopping after you logout
```bash
nohup python uniform_cloud_server.py 2>&1 &
```

## Optional: Setup Server as Systemd Service
Instead of Step 5, you can prefer systemd service
### Enable service
```bash
sudo systemctl enable uniform_cloud_server.service
```

### Start service
```bash
sudo systemctl start uniform_cloud_server.service
```

### Check status of service
```bash
sudo systemctl status uniform_cloud_server.service
```

## Read Logs
### Standart output and error
This will only work if you use Step 5 and ignore systemd service
```bash
tail -f nohup.out
```

### Logging (python module) logs
```bash
tail -f uniform_cloud_server.log
```

### Systemd logs
This will only work if you ignore Step 5 and use systemd service
```bash
sudo journalctl -u uniform_cloud_server.service
```

