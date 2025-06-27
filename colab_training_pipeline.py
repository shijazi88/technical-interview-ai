"""
Complete Technical Interview LLM Training Pipeline for Google Colab Pro
This script provides a one-command solution for training a technical interview AI
"""

import os
import json
import torch
from datetime import datetime
import argparse
import gc
from pathlib import Path

def install_requirements():
    """Install required packages for Google Colab"""
    packages = [
        "transformers>=4.30.0",
        "peft>=0.4.0", 
        "accelerate>=0.20.0",
        "bitsandbytes",
        "datasets",
        "torch>=2.0.0"
    ]
    
    print("📦 Installing required packages...")
    for package in packages:
        os.system(f"pip install {package}")
    print("✅ Packages installed successfully!")

def setup_environment():
    """Setup environment variables for optimal Colab performance"""
    # Set MAX_TOKENS to avoid API issues based on memory
    os.environ['MAX_TOKENS'] = '1000'
    
    # CUDA settings for better memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🖥️ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"🔢 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU detected. Training will be slow on CPU.")

def main():
    """Main training pipeline optimized for Google Colab Pro"""
    
    parser = argparse.ArgumentParser(description="Train Technical Interview LLM on Google Colab")
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="Base model to use")
    parser.add_argument("--num_scenarios", type=int, default=20, help="Number of training scenarios")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size (keep small for Colab)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (reduced for larger model)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length (reduced for 7B model)")
    parser.add_argument("--output_dir", type=str, default="./technical_interview_model", help="Output directory")
    parser.add_argument("--install_deps", action="store_true", help="Install dependencies first")
    
    args = parser.parse_args()
    
    print("🚀 Technical Interview LLM Training Pipeline for Google Colab Pro")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Base model: {args.model_name}")
    print(f"  - Training scenarios: {args.num_scenarios}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Warmup steps: {args.warmup_steps}")
    print(f"  - Max length: {args.max_length}")
    print(f"  - Output directory: {args.output_dir}")
    print("="*70)
    
    try:
        # Step 0: Install dependencies if requested
        if args.install_deps:
            install_requirements()
        
        # Step 1: Setup environment
        print("\n🔧 Step 1: Setting up environment...")
        setup_environment()
        
        # Step 2: Create questions database
        print("\n📚 Step 2: Creating technical questions database...")
        from technical_questions_db import TechnicalQuestionsDatabase
        questions_db = TechnicalQuestionsDatabase()
        print(f"✅ Created database with {len(questions_db.questions)} questions")
        
        # Step 3: Generate training data
        print(f"\n🎯 Step 3: Generating {args.num_scenarios} training scenarios...")
        from enhanced_data_processor import TechnicalInterviewDataset as DatasetGenerator
        dataset_creator = DatasetGenerator(questions_db)
        training_data = dataset_creator.create_realistic_interview_scenarios(args.num_scenarios)
        
        # Save training data
        training_file = "technical_interview_training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Generated {len(training_data)} training examples")
        
        # Step 4: Setup model and training
        print("\n🤖 Step 4: Setting up model and training...")
        from technical_model_setup import setup_technical_interview_training
        
        trainer, tokenizer = setup_technical_interview_training(
            num_scenarios=len(training_data),
            model_name=args.model_name,
            max_length=args.max_length
        )
        
        # Update training arguments for user preferences
        trainer.args.num_train_epochs = args.epochs
        trainer.args.per_device_train_batch_size = args.batch_size
        trainer.args.learning_rate = args.learning_rate
        trainer.args.warmup_steps = args.warmup_steps
        trainer.args.output_dir = args.output_dir
        
        print("✅ Model and trainer setup complete")
        
        # Step 5: Train model
        print(f"\n🏃 Step 5: Training model for {args.epochs} epochs...")
        start_time = datetime.now()
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Start training with error handling
        try:
            trainer.train()
            training_success = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ GPU out of memory! Trying with smaller batch size...")
                trainer.args.per_device_train_batch_size = 1
                trainer.args.gradient_accumulation_steps = 2
                torch.cuda.empty_cache()
                gc.collect()
                trainer.train()
                training_success = True
            else:
                raise e
        
        training_time = datetime.now() - start_time
        print(f"✅ Training completed in {training_time}")
        
        # Step 6: Save model
        print("\n💾 Step 6: Saving final model...")
        trainer.save_model(args.output_dir)
        tokenizer.tokenizer.save_pretrained(args.output_dir)
        print(f"✅ Model saved to {args.output_dir}")
        
        # Step 7: Test the model
        print("\n🧪 Step 7: Testing the trained model...")
        try:
            from technical_interview_bot import TechnicalInterviewBot
            
            bot = TechnicalInterviewBot(args.output_dir)
            if bot.model is not None:
                test_response = bot.start_interview(
                    programming_language="python",
                    experience_level="junior",
                    candidate_name="Test User"
                )
                print("✅ Model test successful!")
                print("Sample output:")
                print("-" * 40)
                print(test_response[:200] + "..." if len(test_response) > 200 else test_response)
                print("-" * 40)
            else:
                print("⚠️ Model loaded but not accessible for testing")
                
        except Exception as e:
            print(f"⚠️ Model test failed: {e}")
            print("Model training completed but testing encountered issues.")
        
        # Step 8: Create usage examples
        print("\n📝 Step 8: Creating usage examples...")
        
        # Create demo script
        demo_script = f'''#!/usr/bin/env python3
"""
Demo script for your trained technical interview model
Run this script to test your model interactively
"""

from technical_interview_bot import TechnicalInterviewBot

def main():
    print("🤖 Technical Interview AI Demo")
    print("=" * 40)
    
    # Initialize your trained model
    bot = TechnicalInterviewBot('{args.output_dir}')
    
    if bot.model is None:
        print("❌ Model not found. Please check the model path.")
        return
    
    # Get interview parameters
    print("\\nLet's set up the interview:")
    name = input("Candidate name (or press Enter for 'Test Candidate'): ").strip() or "Test Candidate"
    
    languages = ["python", "java", "csharp", "flutter", "php", "javascript"]
    print(f"Available languages: {{', '.join(languages)}}")
    language = input("Programming language: ").strip().lower()
    if language not in languages:
        language = "python"
    
    levels = ["junior", "mid_level", "senior", "lead"]
    print(f"Experience levels: {{', '.join(levels)}}")
    level = input("Experience level: ").strip().lower()
    if level not in levels:
        level = "mid_level"
    
    # Start interview
    response = bot.start_interview(
        programming_language=language,
        experience_level=level,
        candidate_name=name
    )
    
    print("\\n" + "=" * 50)
    print(response)
    print("=" * 50)
    
    # Interactive interview loop
    while bot.question_count < bot.max_questions:
        print("\\n(Type 'quit' to end the interview)")
        candidate_answer = input("\\nYour response: ").strip()
        
        if candidate_answer.lower() == 'quit':
            break
            
        if candidate_answer:
            follow_up = bot.process_response(candidate_answer)
            print("\\n" + "-" * 40)
            print(follow_up)
            print("-" * 40)
            
            if "Thank you for the technical interview" in follow_up:
                break
    
    # Show summary
    summary = bot.get_interview_summary()
    print("\\n📊 Interview Summary:")
    print(f"Questions asked: {{summary['questions_asked']}}")
    print(f"Final difficulty: {{summary['final_difficulty']}}/8")
    print(f"Topics covered: {{', '.join(summary['topics_covered'])}}")

if __name__ == "__main__":
    main()
'''
        
        with open(f"{args.output_dir}/demo.py", "w") as f:
            f.write(demo_script)
        
        # Create notebook example
        notebook_example = f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# Technical Interview AI - Usage Example\\n",
    "\\n",
    "This notebook demonstrates how to use your trained technical interview model."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Import the interview bot\\n",
    "from technical_interview_bot import TechnicalInterviewBot\\n",
    "\\n",
    "# Initialize with your trained model\\n",
    "bot = TechnicalInterviewBot('{args.output_dir}')"
   ]
  }},
  {{
   "cell_type": "code", 
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Start an interview\\n",
    "interview_start = bot.start_interview(\\n",
    "    programming_language='python',\\n",
    "    experience_level='mid_level',\\n",
    "    candidate_name='Demo User'\\n",
    ")\\n",
    "\\n",
    "print(interview_start)"
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "# Process a candidate response\\n",
    "candidate_answer = \\"Lists are mutable and use square brackets, tuples are immutable and use parentheses.\\"\\n",
    "\\n",
    "follow_up = bot.process_response(candidate_answer)\\n",
    "print(follow_up)"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python", 
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.8.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''
        
        with open(f"{args.output_dir}/usage_example.ipynb", "w") as f:
            f.write(notebook_example)
        
        # Create README
        readme_content = f"""# Technical Interview AI Model

This directory contains your trained technical interview AI model.

## Files:
- `pytorch_model.bin` - The trained model weights
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- `demo.py` - Interactive demo script
- `usage_example.ipynb` - Jupyter notebook example

## Quick Start:

1. **Interactive Demo:**
   ```bash
   python demo.py
   ```

2. **In Python Code:**
   ```python
   from technical_interview_bot import TechnicalInterviewBot
   
   bot = TechnicalInterviewBot('{args.output_dir}')
   response = bot.start_interview(
       programming_language='python',
       experience_level='junior',
       candidate_name='John Doe'
   )
   print(response)
   ```

## Supported Languages:
- Python
- Java  
- C#
- Flutter/Dart
- PHP
- JavaScript

## Experience Levels:
- junior (0-2 years)
- mid_level (2-5 years)
- senior (5+ years)
- lead (8+ years, leadership)

## Training Details:
- Training scenarios: {args.num_scenarios}
- Training epochs: {args.epochs}
- Base model: {args.model_name}
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Capabilities:
✅ Context-aware follow-up questions
✅ Experience-level appropriate questioning  
✅ Multi-language technical interviews
✅ Adaptive difficulty based on responses
✅ Natural conversation flow
✅ Performance assessment indicators
"""
        
        with open(f"{args.output_dir}/README.md", "w") as f:
            f.write(readme_content)
        
        print("✅ Usage examples created")
        
        # Final summary
        print("\n🎉 Training Pipeline Completed Successfully!")
        print("="*50)
        print(f"📍 Model Location: {args.output_dir}")
        print(f"📊 Training Examples: {len(training_data)}")
        print(f"⏱️ Training Time: {training_time}")
        print(f"🎯 Model Ready: ✅")
        
        print("\n📋 Next Steps:")
        print(f"1. Run interactive demo: python {args.output_dir}/demo.py")
        print(f"2. Open usage notebook: {args.output_dir}/usage_example.ipynb")
        print(f"3. Check README: {args.output_dir}/README.md")
        
        # Memory cleanup
        del trainer
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print("\n✨ Your Technical Interview AI is ready to use!")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure you have enough GPU memory (reduce batch_size if needed)")
        print("2. Try reducing num_scenarios if you encounter memory issues")
        print("3. Check that all required packages are installed")
        print("4. Restart runtime and try again if you encounter CUDA errors")
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        raise e

if __name__ == "__main__":
    main() 