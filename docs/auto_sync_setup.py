#!/usr/bin/env python3
"""
Auto-Sync Setup for Cursor → Colab
Near real-time synchronization options
"""

import subprocess
import os
import json

def create_github_workflow():
    """Create GitHub-based auto-sync workflow"""
    print("🔧 Setting up GitHub Auto-Sync...")
    
    # Create .gitignore if not exists
    gitignore_content = """# Virtual environments
colab_env/
venv/
env/

# Model files (too large)
*.bin
*.safetensors
technical_interview_model/

# Temporary files
.DS_Store
*.tmp
*.log
colab_project.zip
"""
    
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore")
    
    # Create enhanced Colab notebook
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🤖 Technical Interview AI - Auto-Sync\n",
                    "\n",
                    "**Near Real-time Workflow:**\n",
                    "1. ✏️ Edit code in Cursor\n",
                    "2. 🔄 Auto-sync every 30 seconds\n",
                    "3. 🚀 Pull changes in Colab (1 click)\n",
                    "4. 🔥 Train on GPU!\n",
                    "\n",
                    "GitHub repository URL: https://github.com/shijazi88/technical-interview-ai"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 🔄 QUICK SYNC: Get latest changes from Cursor (30 seconds old max!)\n",
                                             "REPO_URL = 'https://github.com/shijazi88/technical-interview-ai'\n",
                    "PROJECT_DIR = 'interview-ai'\n",
                    "\n",
                    "import os\n",
                    "if os.path.exists(PROJECT_DIR):\n",
                    "    print(\"🔄 Pulling latest changes from Cursor...\")\n",
                    "    %cd $PROJECT_DIR\n",
                    "    !git pull origin main\n",
                    "    print(\"✅ Synced! Your latest Cursor code is now here.\")\n",
                    "else:\n",
                    "    print(\"📥 First time: Cloning repository...\")\n",
                    "    !git clone $REPO_URL $PROJECT_DIR\n",
                    "    %cd $PROJECT_DIR\n",
                    "    print(\"✅ Repository cloned!\")\n",
                    "\n",
                    "!ls -la *.py"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 📦 Setup environment\n",
                    "!pip install -q transformers peft accelerate bitsandbytes datasets\n",
                    "\n",
                    "import torch\n",
                    "print(f\"🔥 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None - Enable GPU!'}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 🚀 Train with your latest Cursor code!\n",
                    "!python colab_training_pipeline.py --num_scenarios 100 --epochs 3"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU"
        },
        "nbformat": 4, "nbformat_minor": 4
    }
    
    with open('Auto_Sync_Training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Created Auto_Sync_Training.ipynb")

def create_sync_script():
    """Create simple sync script"""
    
    sync_script = '''#!/bin/bash
# Quick sync script for Cursor → GitHub

echo "🔄 Syncing changes to GitHub..."

# Add all changes
git add .

# Check if there are changes
if git diff --cached --quiet; then
    echo "💤 No changes to sync"
else
    # Commit with timestamp
    git commit -m "Auto-sync: $(date)"
    
    # Push to GitHub
    git push origin main
    
    echo "✅ Synced to GitHub!"
    echo "🔗 Run 'git pull' in Colab to get updates"
fi
'''
    
    with open('quick_sync.sh', 'w') as f:
        f.write(sync_script)
    
    os.chmod('quick_sync.sh', 0o755)
    print("✅ Created quick_sync.sh")

def show_workflows():
    """Show available workflow options"""
    print("\n🎯 AVAILABLE WORKFLOWS")
    print("=" * 60)
    
    print("\n🚀 OPTION 1: GitHub Auto-Sync (30-second delay)")
    print("✏️ Cursor: Edit files")
    print("⚡ Auto: ./quick_sync.sh (or watch script)")
    print("🔄 Colab: !git pull origin main")
    print("🔥 Colab: Train on GPU")
    print("⏱️ Delay: ~30 seconds")
    
    print("\n📂 OPTION 2: Google Drive Sync (5-minute delay)")
    print("✏️ Cursor: Edit files")
    print("☁️ Auto: Google Drive File Stream")
    print("📥 Colab: Mount Drive + copy files")
    print("🔥 Colab: Train on GPU")
    print("⏱️ Delay: ~5 minutes")
    
    print("\n🔗 OPTION 3: VS Code Remote (Real-time)")
    print("✏️ Cursor: Edit files")
    print("🌐 SSH: Connect to cloud instance")
    print("🔄 Real-time: No sync delay")
    print("🔥 Cloud: Train on remote GPU")
    print("⏱️ Delay: Real-time!")

def main():
    """Main setup function"""
    print("🎯 CURSOR → COLAB AUTO-SYNC SETUP")
    print("=" * 50)
    
    # Create GitHub workflow
    create_github_workflow()
    create_sync_script()
    
    # Show workflows
    show_workflows()
    
    print("\n📋 RECOMMENDED SETUP (GitHub):")
    print("1. Create GitHub repository")
    print("2. git remote add origin https://github.com/shijazi88/technical-interview-ai")
    print("3. git push -u origin main")
    print("4. Edit in Cursor, run ./quick_sync.sh")
    print("5. In Colab: !git pull origin main")
    print("6. Train with latest code!")
    
    print("\n⚡ QUICK START:")
    print("./quick_sync.sh      # Sync changes to GitHub")
    print("# In Colab: !git pull # Get latest changes")

if __name__ == "__main__":
    main() 