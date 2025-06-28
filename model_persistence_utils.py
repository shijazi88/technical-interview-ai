#!/usr/bin/env python3
"""
Model Persistence Utilities
Comprehensive backup and download strategies for trained models
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import json

class ModelPersistenceManager:
    """Manages model saving, backup, and download strategies"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def mount_google_drive(self):
        """Mount Google Drive for persistent storage"""
        try:
            from google.colab import drive
            print("📂 Mounting Google Drive...")
            drive.mount('/content/drive')
            
            backup_dir = "/content/drive/MyDrive/Technical_Interview_Models"
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Drive mounted. Backup directory: {backup_dir}")
            return backup_dir
        except ImportError:
            print("⚠️ Google Drive mounting only available in Colab")
            return None
        except Exception as e:
            print(f"❌ Drive mounting failed: {e}")
            return None
    
    def backup_to_drive(self, backup_dir: str):
        """Backup model to Google Drive"""
        if not backup_dir:
            return False
        
        try:
            print("💾 Backing up to Google Drive...")
            
            # Create timestamped backup folder
            drive_model_dir = f"{backup_dir}/model_{self.timestamp}"
            shutil.copytree(self.model_dir, drive_model_dir)
            print(f"✅ Model copied to: {drive_model_dir}")
            
            # Create ZIP for easier download
            zip_path = f"{backup_dir}/technical_interview_model_{self.timestamp}.zip"
            self._create_model_zip(zip_path)
            print(f"✅ ZIP backup created: {zip_path}")
            
            # Create download script
            self._create_download_script(backup_dir)
            
            return True
            
        except Exception as e:
            print(f"❌ Drive backup failed: {e}")
            return False
    
    def upload_to_huggingface(self, repo_name: str, token: str = None):
        """Upload model to Hugging Face Hub"""
        try:
            from huggingface_hub import HfApi, login, create_repo
            
            print(f"🤗 Uploading to Hugging Face Hub: {repo_name}")
            
            # Login if token provided
            if token:
                login(token=token)
            
            # Create repository
            api = HfApi()
            try:
                create_repo(repo_id=repo_name, exist_ok=True)
            except Exception as e:
                print(f"⚠️ Repo creation note: {e}")
            
            # Upload model files
            api.upload_folder(
                folder_path=self.model_dir,
                repo_id=repo_name,
                repo_type="model"
            )
            
            print(f"✅ Model uploaded to: https://huggingface.co/{repo_name}")
            
            # Create model card
            self._create_model_card(repo_name)
            
            return True
            
        except ImportError:
            print("❌ huggingface_hub not installed")
            return False
        except Exception as e:
            print(f"❌ HF upload failed: {e}")
            print(f"💡 Manual upload: huggingface-cli upload {repo_name} {self.model_dir}")
            return False
    
    def create_local_download_package(self):
        """Create a downloadable ZIP package"""
        try:
            zip_path = f"technical_interview_model_{self.timestamp}.zip"
            self._create_model_zip(zip_path)
            
            print(f"✅ Download package created: {zip_path}")
            
            # Create download script for Colab
            download_script = f"""
# Download your trained model from Colab
from google.colab import files
files.download('{zip_path}')
"""
            
            with open("download_model.py", "w") as f:
                f.write(download_script)
            
            print("💡 Run 'python download_model.py' to download model")
            return zip_path
            
        except Exception as e:
            print(f"❌ Package creation failed: {e}")
            return None
    
    def _create_model_zip(self, zip_path: str):
        """Create ZIP archive of the model"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.model_dir)
                    zipf.write(file_path, arcname)
    
    def _create_download_script(self, backup_dir: str):
        """Create download instructions script"""
        script_content = f'''
# Technical Interview Model Download Instructions
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Method 1: Download from Google Drive
1. Open Google Drive
2. Navigate to: Technical_Interview_Models
3. Download: technical_interview_model_{self.timestamp}.zip
4. Extract and use locally

## Method 2: Direct Colab Download (while session active)
```python
# Run in Colab cell:
from google.colab import files
files.download('technical_interview_model_{self.timestamp}.zip')
```

## Method 3: Clone from Google Drive (programmatic)
```python
import shutil
model_backup = '{backup_dir}/model_{self.timestamp}'
local_model = './downloaded_interview_model'
shutil.copytree(model_backup, local_model)
```

## Usage after download:
```python
from technical_interview_bot import TechnicalInterviewBot

# Point to your downloaded model
bot = TechnicalInterviewBot('./path/to/extracted/model')

# Test it
response = bot.start_interview(
    programming_language='python',
    experience_level='mid_level',
    candidate_name='Test User'
)
print(response)
```

## Model Information:
- Training Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Model Directory: {self.model_dir}
- Backup Location: {backup_dir}/model_{self.timestamp}
- ZIP Location: {backup_dir}/technical_interview_model_{self.timestamp}.zip
'''
        
        instructions_path = f"{backup_dir}/DOWNLOAD_INSTRUCTIONS_{self.timestamp}.md"
        with open(instructions_path, "w") as f:
            f.write(script_content)
        
        print(f"✅ Download instructions saved: {instructions_path}")
    
    def _create_model_card(self, repo_name: str):
        """Create model card for Hugging Face"""
        model_card = f"""---
license: apache-2.0
base_model: codellama/CodeLlama-7b-Instruct-hf
tags:
- technical-interview
- code-generation
- interview-ai
- llama
- peft
- lora
language:
- en
pipeline_tag: text-generation
---

# Technical Interview AI - CodeLlama Fine-tuned

This model is a fine-tuned version of CodeLlama-7B specifically trained for conducting technical interviews across multiple programming languages.

## Model Description

- **Base Model**: CodeLlama-7b-Instruct-hf
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Use Case**: Technical interview simulation and assessment

## Supported Languages

- Python
- Java
- C#
- Flutter/Dart
- PHP
- JavaScript

## Experience Levels

- Junior (0-2 years)
- Mid-level (2-5 years)
- Senior (5+ years)
- Lead (8+ years)

## Usage

```python
from technical_interview_bot import TechnicalInterviewBot

# Load the model
bot = TechnicalInterviewBot("./model_directory")

# Start an interview
response = bot.start_interview(
    programming_language="python",
    experience_level="mid_level",
    candidate_name="John Doe"
)

print(response)
```

## Training Details

The model was fine-tuned on a dataset of realistic technical interview scenarios, including:
- Context-aware question progression
- Experience-level appropriate difficulty
- Multi-language technical concepts
- Natural conversation flow

## Limitations

- Designed specifically for technical interviews
- May not perform well on general conversation
- Requires technical_interview_bot wrapper for optimal performance

## Repository

Model repository: {repo_name}
"""
        
        card_path = f"{self.model_dir}/README.md"
        with open(card_path, "w") as f:
            f.write(model_card)

def quick_backup_all(model_dir: str, hf_repo_name: str = None, hf_token: str = None):
    """Quick function to backup model with all strategies"""
    print("🚀 Starting comprehensive model backup...")
    
    manager = ModelPersistenceManager(model_dir)
    
    # Strategy 1: Google Drive
    backup_dir = manager.mount_google_drive()
    if backup_dir:
        manager.backup_to_drive(backup_dir)
    
    # Strategy 2: Hugging Face
    if hf_repo_name:
        manager.upload_to_huggingface(hf_repo_name, hf_token)
    
    # Strategy 3: Local download package
    manager.create_local_download_package()
    
    print("✅ All backup strategies completed!")
    
    return {
        'drive_backup': backup_dir is not None,
        'hf_upload': hf_repo_name is not None,
        'local_package': True
    }

# Example usage for Colab
def colab_save_model(model_dir: str):
    """Simple function for Colab users to save their model"""
    print("💾 Saving your trained model with multiple backup strategies...")
    
    # Get user preferences
    print("\n📋 Backup options:")
    print("1. Google Drive backup (recommended)")
    print("2. Hugging Face Hub upload (optional)")
    print("3. Local download package (always included)")
    
    # Always try Drive backup
    manager = ModelPersistenceManager(model_dir)
    backup_dir = manager.mount_google_drive()
    if backup_dir:
        manager.backup_to_drive(backup_dir)
    
    # Optional HF upload
    hf_upload = input("\n❓ Upload to Hugging Face Hub? (y/n): ").strip().lower()
    if hf_upload == 'y':
        repo_name = input("🤗 Enter repo name (username/model-name): ").strip()
        if repo_name:
            print("💡 You'll need to login to HF Hub if not already logged in")
            manager.upload_to_huggingface(repo_name)
    
    # Always create download package
    manager.create_local_download_package()
    
    print("\n🎉 Model saved successfully!")
    print("📋 Your model is now preserved with multiple backup strategies")

if __name__ == "__main__":
    # Example usage
    model_directory = "./technical_interview_model"
    colab_save_model(model_directory) 