#!/usr/bin/env python3
"""
GitHub Auto-Sync for Cursor → Colab
Near real-time synchronization using GitHub as bridge
"""

import subprocess
import os
import time
import json
from pathlib import Path
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GitAutoSync:
    """Automatic Git synchronization for Cursor → Colab workflow"""
    
    def __init__(self, repo_url=None):
        self.repo_url = repo_url
        self.auto_commit = True
        self.sync_interval = 30  # seconds
        
    def setup_repository(self):
        """Initialize Git repository if not exists"""
        if not os.path.exists('.git'):
            print("🔧 Setting up Git repository...")
            
            # Initialize git
            subprocess.run(['git', 'init'], check=True)
            
            # Create .gitignore
            gitignore_content = """
# Virtual environments
colab_env/
venv/
env/

# Compiled files
__pycache__/
*.pyc
*.pyo

# Model files (too large for git)
*.bin
*.safetensors
technical_interview_model/

# Temporary files
.DS_Store
*.tmp
*.log

# Zip packages (recreated automatically)
colab_project.zip
trained_interview_ai.zip
"""
            with open('.gitignore', 'w') as f:
                f.write(gitignore_content)
            
            # Add files
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit: Technical Interview AI'], check=True)
            
            print("✅ Git repository initialized")
        else:
            print("✅ Git repository already exists")
    
    def create_colab_notebook_with_github(self):
        """Create enhanced Colab notebook that pulls from GitHub"""
        
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# 🤖 Technical Interview AI - GitHub Auto-Sync\n",
                        "\n",
                        "**Real-time Workflow:**\n",
                        "1. ✏️ Edit code in Cursor\n",
                        "2. 🔄 Auto-sync via GitHub (30 seconds)\n",
                        "3. 🚀 Pull latest changes in Colab\n",
                        "4. 🔥 Train on GPU!\n",
                        "\n",
                        "**GitHub Repository:** https://github.com/shijazi88/technical-interview-ai"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🔄 STEP 1: Clone/Update from GitHub (Run this first!)\n",
                        "import os\n",
                        "import subprocess\n",
                        "\n",
                        "REPO_URL = 'https://github.com/shijazi88/technical-interview-ai'\n",
                        "PROJECT_DIR = 'technical-interview-ai'\n",
                        "\n",
                        "if os.path.exists(PROJECT_DIR):\n",
                        "    print(\"🔄 Pulling latest changes...\")\n",
                        "    os.chdir(PROJECT_DIR)\n",
                        "    !git pull origin main\n",
                        "    print(\"✅ Updated to latest version from Cursor!\")\n",
                        "else:\n",
                        "    print(\"📥 Cloning repository...\")\n",
                        "    !git clone $REPO_URL $PROJECT_DIR\n",
                        "    os.chdir(PROJECT_DIR)\n",
                        "    print(\"✅ Repository cloned!\")\n",
                        "\n",
                        "print(\"📁 Available files:\")\n",
                        "!ls -la *.py"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🔄 QUICK UPDATE: Run this cell to get latest changes from Cursor\n",
                        "print(\"🔄 Getting latest updates...\")\n",
                        "!git pull origin main\n",
                        "print(\"✅ Synchronized with Cursor!\")\n",
                        "\n",
                        "# Show what changed\n",
                        "!git log --oneline -5"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 📦 Install packages\n",
                        "!pip install transformers>=4.30.0 peft>=0.4.0 accelerate>=0.20.0 bitsandbytes datasets\n",
                        "\n",
                        "# Check GPU\n",
                        "import torch\n",
                        "if torch.cuda.is_available():\n",
                        "    print(f\"🔥 GPU: {torch.cuda.get_device_name(0)}\")\n",
                        "else:\n",
                        "    print(\"⚠️ No GPU - Enable GPU in Runtime > Change runtime type\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🚀 Train your AI with latest code from Cursor!\n",
                        "!python colab_training_pipeline.py --num_scenarios 150 --epochs 3\n",
                        "\n",
                        "print(\"🎉 Training completed with your latest Cursor code!\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 🧪 Test the trained model\n",
                        "from technical_interview_bot import TechnicalInterviewBot\n",
                        "\n",
                        "bot = TechnicalInterviewBot('./technical_interview_model')\n",
                        "response = bot.start_interview(\n",
                        "    programming_language='python',\n",
                        "    experience_level='mid_level',\n",
                        "    candidate_name='GitHub Test'\n",
                        ")\n",
                        "print(response)"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 💾 Package trained model for download\n",
                        "import shutil\n",
                        "shutil.make_archive('trained_model', 'zip', 'technical_interview_model')\n",
                        "print(\"✅ Download 'trained_model.zip' from Files panel!\")"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.12"},
                "accelerator": "GPU"
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        with open('GitHub_Colab_Training.ipynb', 'w') as f:
            json.dump(notebook, f, indent=2)
        
        print("✅ Created GitHub_Colab_Training.ipynb")

class FileWatcher(FileSystemEventHandler):
    """Watch for file changes and auto-commit"""
    
    def __init__(self, git_sync):
        self.git_sync = git_sync
        self.last_commit = 0
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only watch Python files
        if event.src_path.endswith('.py'):
            current_time = time.time()
            
            # Debounce - only commit every 30 seconds
            if current_time - self.last_commit > 30:
                self.last_commit = current_time
                self.git_sync.auto_commit_changes(f"Auto-update: {Path(event.src_path).name}")

class GitAutoSync:
    """Enhanced GitAutoSync with file watching"""
    
    def __init__(self):
        self.watching = False
        
    def auto_commit_changes(self, message="Auto-update from Cursor"):
        """Automatically commit and push changes"""
        try:
            # Add all changed files
            subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
            
            # Check if there are changes to commit
            result = subprocess.run(['git', 'diff', '--cached'], capture_output=True, text=True)
            if result.stdout.strip():
                # Commit changes
                subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True)
                
                # Push to GitHub
                subprocess.run(['git', 'push', 'origin', 'main'], check=True, capture_output=True)
                
                print(f"✅ Auto-synced: {message}")
            else:
                print("💤 No changes to sync")
                
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Sync failed: {e}")
    
    def start_watching(self):
        """Start watching files for changes"""
        print("👀 Starting file watcher...")
        print("🔄 Files will auto-sync to GitHub every 30 seconds when changed")
        print("⏹️ Press Ctrl+C to stop watching")
        
        event_handler = FileWatcher(self)
        observer = Observer()
        observer.schedule(event_handler, '.', recursive=False)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n🛑 File watching stopped")
        
        observer.join()

def setup_github_workflow():
    """Complete GitHub workflow setup"""
    print("🔧 Setting up GitHub Auto-Sync workflow...")
    
    sync = GitAutoSync()
    
    # Setup repository
    sync.setup_repository()
    
    # Create enhanced notebook
    sync.create_colab_notebook_with_github()
    
    print("\n🎯 GITHUB AUTO-SYNC WORKFLOW")
    print("=" * 50)
    print("✏️  1. EDIT IN CURSOR:")
    print("   - Make changes to any .py file")
    print("   - Files auto-sync every 30 seconds")
    print()
    print("🔄 2. AUTO-SYNC TO GITHUB:")
    print("   - Changes committed automatically")
    print("   - Pushed to your GitHub repository")
    print()
    print("🚀 3. UPDATE IN COLAB:")
    print("   - Run: !git pull origin main")
    print("   - Get latest changes instantly")
    print("   - Train with updated code!")
    print()
    print("📋 SETUP STEPS:")
    print("1. Create GitHub repository")
    print("2. Add remote: git remote add origin https://github.com/shijazi88/technical-interview-ai")
    print("3. Push initial code: git push -u origin main")
    print("4. Start file watcher: python github_auto_sync.py --watch")
    print("5. Use GitHub_Colab_Training.ipynb in Colab")
    
    return sync

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--watch':
        # Start file watching mode
        sync = GitAutoSync()
        sync.start_watching()
    else:
        # Setup mode
        setup_github_workflow() 