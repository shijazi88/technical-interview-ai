"""
Cursor + Google Colab Integration Script
This script provides multiple ways to connect Cursor with Google Colab
"""

import os
import subprocess
import json
import time
from pathlib import Path

class ColabCursorIntegration:
    """Integration between Cursor and Google Colab"""
    
    def __init__(self):
        self.colab_token = None
        self.project_path = Path.cwd()
        
    def setup_colab_local_runtime(self):
        """Setup local runtime connection to Colab"""
        print("🔧 Setting up Colab Local Runtime Connection...")
        
        setup_code = '''
# Run this in Google Colab to connect to local runtime:

# 1. Install required packages in Colab
!pip install jupyter_http_over_ws
!jupyter serverextension enable --py jupyter_http_over_ws

# 2. Start local Jupyter server (run this in Cursor terminal)
# jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0

# 3. In Colab, click "Connect to local runtime" and use: http://localhost:8888/
'''
        
        with open('colab_local_setup.py', 'w') as f:
            f.write(setup_code)
        
        print("✅ Setup code saved to 'colab_local_setup.py'")
        return setup_code
    
    def create_sync_script(self):
        """Create file synchronization script"""
        print("📁 Creating file sync script...")
        
        sync_script = '''#!/bin/bash
# File synchronization script for Cursor <-> Colab

# Upload files to Colab (using Google Drive)
upload_to_colab() {
    echo "🔄 Uploading files to Google Drive for Colab access..."
    
    # Create a compressed archive
    tar -czf project_files.tar.gz *.py *.json *.md
    
    # Upload using gdrive (install: https://github.com/gdrive-org/gdrive)
    # gdrive upload project_files.tar.gz
    
    echo "✅ Files uploaded to Google Drive"
    echo "📝 In Colab, use: !tar -xzf /content/drive/MyDrive/project_files.tar.gz"
}

# Download results from Colab
download_from_colab() {
    echo "⬇️ Downloading results from Colab..."
    # Implementation depends on your sharing method
}

# Real-time sync using rsync (if you have SSH access)
sync_with_ssh() {
    echo "🔄 Syncing with remote server..."
    # rsync -avz . user@server:/path/to/project/
}

# Usage examples
case "$1" in
    upload)
        upload_to_colab
        ;;
    download)
        download_from_colab
        ;;
    ssh)
        sync_with_ssh
        ;;
    *)
        echo "Usage: $0 {upload|download|ssh}"
        exit 1
        ;;
esac
'''
        
        with open('sync_colab.sh', 'w') as f:
            f.write(sync_script)
        
        os.chmod('sync_colab.sh', 0o755)
        print("✅ Sync script created: './sync_colab.sh'")
    
    def create_colab_api_client(self):
        """Create a simple Colab API client"""
        print("🌐 Creating Colab API client...")
        
        api_client = '''
import requests
import json
from typing import Dict, Any

class ColabAPIClient:
    """Simple client for interacting with Google Colab"""
    
    def __init__(self, auth_token: str = None):
        self.auth_token = auth_token
        self.base_url = "https://colab.research.google.com"
        self.session = requests.Session()
        
        if auth_token:
            self.session.headers.update({
                'Authorization': f'Bearer {auth_token}'
            })
    
    def execute_code(self, notebook_id: str, code: str) -> Dict[str, Any]:
        """Execute code in a Colab notebook"""
        # Note: This is a simplified example
        # Actual Colab API usage requires proper authentication
        
        payload = {
            'code': code,
            'notebook_id': notebook_id
        }
        
        # This would need proper Colab API endpoints
        response = self.session.post(
            f"{self.base_url}/api/execute", 
            json=payload
        )
        
        return response.json()
    
    def upload_file(self, file_path: str) -> str:
        """Upload file to Colab environment"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/api/upload",
                files=files
            )
        
        return response.json().get('file_id')
    
    def create_notebook_from_script(self, script_path: str) -> str:
        """Convert Python script to Colab notebook"""
        with open(script_path, 'r') as f:
            code = f.read()
        
        # Convert to notebook format
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": code.split('\\n'),
                    "metadata": {},
                    "execution_count": None,
                    "outputs": []
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook

# Usage example
if __name__ == "__main__":
    client = ColabAPIClient()
    
    # Convert script to notebook
    notebook = client.create_notebook_from_script('colab_training_pipeline.py')
    
    # Save as .ipynb file
    with open('auto_generated_notebook.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Notebook created: auto_generated_notebook.ipynb")
'''
        
        with open('colab_api_client.py', 'w') as f:
            f.write(api_client)
        
        print("✅ API client created: 'colab_api_client.py'")
    
    def create_cursor_extension_config(self):
        """Create configuration for Cursor extensions"""
        print("🔌 Creating Cursor extension configuration...")
        
        # VS Code settings that work with Cursor
        vscode_settings = {
            "python.defaultInterpreterPath": "/usr/bin/python3",
            "jupyter.notebookFileRoot": "${workspaceFolder}",
            "jupyter.alwaysTrustNotebooks": True,
            "files.associations": {
                "*.ipynb": "jupyter-notebook"
            },
            "jupyter.sendSelectionToInteractiveWindow": True,
            "jupyter.interactiveWindowMode": "perFile",
            "remote.SSH.remotePlatform": {
                "colab-server": "linux"
            }
        }
        
        os.makedirs('.vscode', exist_ok=True)
        with open('.vscode/settings.json', 'w') as f:
            json.dump(vscode_settings, f, indent=2)
        
        # Create tasks for common operations
        tasks = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "Upload to Colab",
                    "type": "shell",
                    "command": "./sync_colab.sh upload",
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Start Local Jupyter",
                    "type": "shell",
                    "command": "jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0",
                    "group": "build",
                    "presentation": {
                        "echo": True,
                        "reveal": "always",
                        "focus": False,
                        "panel": "shared"
                    }
                },
                {
                    "label": "Convert to Notebook",
                    "type": "shell",
                    "command": "python colab_api_client.py",
                    "group": "build"
                }
            ]
        }
        
        with open('.vscode/tasks.json', 'w') as f:
            json.dump(tasks, f, indent=2)
        
        print("✅ Cursor/VS Code configuration created")
    
    def setup_remote_development(self):
        """Setup remote development configuration"""
        print("🌍 Setting up remote development...")
        
        remote_config = '''
# SSH Config for remote development
# Add to ~/.ssh/config

Host colab-server
    HostName your-colab-server.com
    User your-username
    Port 22
    IdentityFile ~/.ssh/id_rsa
    ForwardX11 yes
    
# Port forwarding for Jupyter
Host colab-jupyter
    HostName localhost
    Port 8888
    LocalForward 8888 localhost:8888
'''
        
        with open('ssh_config_example.txt', 'w') as f:
            f.write(remote_config)
        
        print("✅ SSH config example created")
    
    def create_workflow_scripts(self):
        """Create workflow automation scripts"""
        print("⚡ Creating workflow automation...")
        
        workflow_script = '''#!/usr/bin/env python3
"""
Automated workflow for Cursor + Colab development
"""

import os
import subprocess
import time
import webbrowser
from pathlib import Path

class ColabWorkflow:
    def __init__(self):
        self.project_root = Path.cwd()
        
    def start_local_jupyter(self):
        """Start local Jupyter server for Colab connection"""
        print("🚀 Starting local Jupyter server...")
        
        cmd = [
            "jupyter", "notebook",
            "--NotebookApp.allow_origin=https://colab.research.google.com",
            "--port=8888",
            "--NotebookApp.port_retries=0",
            "--no-browser"
        ]
        
        try:
            process = subprocess.Popen(cmd)
            print("✅ Jupyter server started on http://localhost:8888")
            print("🔗 Connect in Colab: Runtime > Connect to local runtime")
            return process
        except Exception as e:
            print(f"❌ Failed to start Jupyter: {e}")
            return None
    
    def sync_files_to_drive(self):
        """Sync project files to Google Drive"""
        print("📁 Syncing files to Google Drive...")
        
        # Create archive of important files
        files_to_sync = [
            "*.py", "*.json", "*.md", "*.txt", "*.yml", "*.yaml"
        ]
        
        archive_cmd = ["tar", "-czf", "project_sync.tar.gz"] + files_to_sync
        
        try:
            subprocess.run(archive_cmd, check=True)
            print("✅ Project files archived")
            
            # Instructions for manual upload
            print("📝 Upload 'project_sync.tar.gz' to Google Drive")
            print("📝 In Colab: !tar -xzf /content/drive/MyDrive/project_sync.tar.gz")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create archive: {e}")
    
    def convert_script_to_notebook(self, script_path: str):
        """Convert Python script to Jupyter notebook"""
        print(f"📓 Converting {script_path} to notebook...")
        
        try:
            import nbformat
            from nbformat.v4 import new_notebook, new_code_cell
            
            with open(script_path, 'r') as f:
                code = f.read()
            
            # Split code into logical cells (by functions/classes)
            cells = []
            current_cell = []
            
            for line in code.split('\\n'):
                if (line.startswith('def ') or line.startswith('class ') or 
                    line.startswith('# %%') or line.startswith('"""')):
                    if current_cell:
                        cells.append(new_code_cell('\\n'.join(current_cell)))
                        current_cell = []
                current_cell.append(line)
            
            if current_cell:
                cells.append(new_code_cell('\\n'.join(current_cell)))
            
            notebook = new_notebook(cells=cells)
            
            output_path = script_path.replace('.py', '.ipynb')
            with open(output_path, 'w') as f:
                nbformat.write(notebook, f)
            
            print(f"✅ Notebook created: {output_path}")
            return output_path
            
        except ImportError:
            print("❌ nbformat not installed. Run: pip install nbformat")
            return None
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            return None
    
    def open_colab_with_notebook(self, notebook_path: str):
        """Open notebook in Google Colab"""
        print(f"🌐 Opening {notebook_path} in Colab...")
        
        # Upload to GitHub/Drive first, then open in Colab
        colab_url = f"https://colab.research.google.com/github/your-username/your-repo/blob/main/{notebook_path}"
        
        # Or use local file upload
        colab_upload_url = "https://colab.research.google.com/notebooks/intro.ipynb#file_upload"
        
        webbrowser.open(colab_upload_url)
        print("📝 Upload your notebook file in the Colab interface")

if __name__ == "__main__":
    import sys
    
    workflow = ColabWorkflow()
    
    if len(sys.argv) < 2:
        print("Usage: python workflow.py [jupyter|sync|convert|open] [file]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "jupyter":
        workflow.start_local_jupyter()
    elif command == "sync":
        workflow.sync_files_to_drive()
    elif command == "convert":
        if len(sys.argv) < 3:
            print("Please specify script file to convert")
            sys.exit(1)
        workflow.convert_script_to_notebook(sys.argv[2])
    elif command == "open":
        if len(sys.argv) < 3:
            print("Please specify notebook file to open")
            sys.exit(1)
        workflow.open_colab_with_notebook(sys.argv[2])
    else:
        print("Unknown command. Use: jupyter, sync, convert, or open")
'''
        
        with open('colab_workflow.py', 'w') as f:
            f.write(workflow_script)
        
        os.chmod('colab_workflow.py', 0o755)
        print("✅ Workflow script created: 'colab_workflow.py'")

def main():
    """Setup Cursor + Colab integration"""
    print("🚀 Setting up Cursor + Google Colab Integration")
    print("=" * 50)
    
    integration = ColabCursorIntegration()
    
    # Create all integration components
    integration.setup_colab_local_runtime()
    integration.create_sync_script()
    integration.create_colab_api_client()
    integration.create_cursor_extension_config()
    integration.setup_remote_development()
    integration.create_workflow_scripts()
    
    print("\n🎉 Integration setup complete!")
    print("\n📋 Quick Start Guide:")
    print("1. Start local Jupyter: python colab_workflow.py jupyter")
    print("2. Connect Colab to local runtime")
    print("3. Sync files: ./sync_colab.sh upload")
    print("4. Convert scripts: python colab_workflow.py convert script.py")
    
    print("\n🔧 Available Commands:")
    print("- python colab_workflow.py jupyter    # Start local Jupyter server")
    print("- python colab_workflow.py sync       # Sync files to Drive")
    print("- python colab_workflow.py convert    # Convert .py to .ipynb")
    print("- ./sync_colab.sh upload              # Upload project files")

if __name__ == "__main__":
    main() 