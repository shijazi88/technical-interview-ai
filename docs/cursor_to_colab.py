#!/usr/bin/env python3
"""
Cursor to Google Colab Workflow
Write code in Cursor, execute on Colab's GPUs
"""

import os
import shutil
import zipfile
from pathlib import Path
import json

def create_colab_package():
    """Package project files for Google Colab upload"""
    print("📦 Creating Colab package...")
    
    # Files to include
    files_to_sync = [
        "technical_questions_db.py",
        "enhanced_data_processor.py", 
        "technical_model_setup.py",
        "technical_interview_bot.py",
        "colab_training_pipeline.py",
        "README.md"
    ]
    
    # Create a zip file for easy upload to Colab
    with zipfile.ZipFile('colab_project.zip', 'w') as zipf:
        for file in files_to_sync:
            if os.path.exists(file):
                zipf.write(file)
                print(f"  ✅ Added {file}")
            else:
                print(f"  ⚠️ Missing {file}")
    
    print("✅ Created colab_project.zip")
    return "colab_project.zip"

def create_colab_notebook():
    """Create a Colab notebook with your project"""
    
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🤖 Technical Interview AI Training\n",
                    "\n",
                    "This notebook trains your AI technical interviewer using files from Cursor.\n",
                    "\n",
                    "**Workflow:**\n",
                    "1. ✏️ Write/edit code in Cursor\n",
                    "2. 📦 Upload project files to Colab\n",
                    "3. 🚀 Train on Colab's GPUs\n",
                    "4. 💾 Download results back to Cursor"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 1: Upload your colab_project.zip file\n",
                    "# Go to Files panel (📁) and upload colab_project.zip\n",
                    "\n",
                    "# Step 2: Extract project files\n",
                    "import zipfile\n",
                    "import os\n",
                    "\n",
                    "# Extract the uploaded project\n",
                    "with zipfile.ZipFile('colab_project.zip', 'r') as zip_ref:\n",
                    "    zip_ref.extractall('.')\n",
                    "\n",
                    "print(\"✅ Project files extracted!\")\n",
                    "print(\"📁 Available files:\")\n",
                    "for file in os.listdir('.'):\n",
                    "    if file.endswith('.py'):\n",
                    "        print(f\"  - {file}\")"
                ]
            },
            {
                "cell_type": "code", 
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 3: Install required packages\n",
                    "!pip install transformers>=4.30.0\n",
                    "!pip install peft>=0.4.0\n",
                    "!pip install accelerate>=0.20.0\n",
                    "!pip install bitsandbytes\n",
                    "!pip install datasets\n",
                    "\n",
                    "print(\"✅ Packages installed!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 4: Check GPU availability\n",
                    "import torch\n",
                    "\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f\"🔥 GPU Available: {torch.cuda.get_device_name(0)}\")\n",
                    "    print(f\"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")\n",
                    "else:\n",
                    "    print(\"⚠️ No GPU detected. Go to Runtime > Change runtime type > GPU\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 5: Train your Technical Interview AI!\n",
                    "# This runs the code you wrote in Cursor on Colab's GPUs\n",
                    "\n",
                    "!python colab_training_pipeline.py --num_scenarios 150 --epochs 3\n",
                    "\n",
                    "print(\"🎉 Training completed on Colab's GPUs!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 6: Test your trained model\n",
                    "from technical_interview_bot import TechnicalInterviewBot\n",
                    "\n",
                    "# Load your trained model\n",
                    "bot = TechnicalInterviewBot('./technical_interview_model')\n",
                    "\n",
                    "# Test it\n",
                    "response = bot.start_interview(\n",
                    "    programming_language='python',\n",
                    "    experience_level='mid_level',\n",
                    "    candidate_name='Test User'\n",
                    ")\n",
                    "\n",
                    "print(response)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 7: Download your trained model\n",
                    "# Package the trained model for download\n",
                    "\n",
                    "import shutil\n",
                    "\n",
                    "# Create archive of trained model\n",
                    "shutil.make_archive('trained_interview_ai', 'zip', 'technical_interview_model')\n",
                    "\n",
                    "print(\"✅ Model packaged as 'trained_interview_ai.zip'\")\n",
                    "print(\"📥 Download it from the Files panel to use in Cursor!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "accelerator": "GPU",
            "gpuClass": "standard"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebook
    with open('Technical_Interview_Training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("✅ Created Technical_Interview_Training.ipynb")
    return "Technical_Interview_Training.ipynb"

def show_workflow():
    """Show the complete Cursor → Colab workflow"""
    print("🎯 CURSOR → COLAB WORKFLOW")
    print("=" * 50)
    print("✏️  1. WRITE CODE IN CURSOR:")
    print("   - Edit Python files in Cursor")
    print("   - Use Cursor's AI assistance")
    print("   - Superior debugging and intellisense")
    print()
    print("📦 2. PACKAGE FOR COLAB:")
    print("   - Run: python cursor_to_colab.py")
    print("   - Creates colab_project.zip")
    print()
    print("🚀 3. EXECUTE ON COLAB:")
    print("   - Upload colab_project.zip to Google Colab")
    print("   - Run training on Colab's GPUs")
    print("   - Monitor progress in Colab")
    print()
    print("💾 4. DOWNLOAD RESULTS:")
    print("   - Download trained model")
    print("   - Use in Cursor for further development")
    print()
    print("🔄 REPEAT: Edit in Cursor → Upload → Train → Download")

if __name__ == "__main__":
    print("🎯 Setting up Cursor → Colab workflow...")
    print()
    
    # Create package and notebook
    zip_file = create_colab_package()
    notebook_file = create_colab_notebook()
    
    print()
    show_workflow()
    
    print()
    print("📋 NEXT STEPS:")
    print("1. Upload 'colab_project.zip' to Google Colab")
    print("2. Open 'Technical_Interview_Training.ipynb' in Colab")
    print("3. Run the cells to train on Colab's GPUs!")
    print()
    print("🔗 Go to: https://colab.research.google.com") 