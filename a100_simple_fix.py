#!/usr/bin/env python3
"""
A100 Simple Training - FlashAttention-Free
Complete training script that bypasses FlashAttention import issues
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
import gc

# CRITICAL: Set environment variables BEFORE any transformers imports
os.environ['TRANSFORMERS_NO_FLASH_ATTN'] = '1'
os.environ['DISABLE_FLASH_ATTN'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def check_gpu():
    """Check and report GPU status"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"🖥️ GPU: {gpu_name}")
        print(f"🔢 Memory: {gpu_memory:.1f} GB")
        
        a100_available = "A100" in gpu_name
        if a100_available:
            print("🎉 A100 DETECTED! Optimizing for maximum performance...")
        else:
            print("⚠️ T4 detected - training will be slower but still works")
        
        return a100_available
    else:
        print("❌ No GPU detected!")
        return False

def import_transformers_safely():
    """Import transformers with FlashAttention completely disabled"""
    try:
        # Try to remove any FlashAttention references
        import sys
        flash_modules = [module for module in sys.modules.keys() if 'flash' in module.lower()]
        for module in flash_modules:
            if module in sys.modules:
                del sys.modules[module]
        
        # Now import transformers
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
            BitsAndBytesConfig
        )
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        import datasets
        from datasets import Dataset
        
        print("✅ Transformers imported successfully (FlashAttention bypassed)")
        return True, (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, 
                     DataCollatorForLanguageModeling, BitsAndBytesConfig, LoraConfig, 
                     get_peft_model, TaskType, prepare_model_for_kbit_training, Dataset)
    
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False, None

def setup_model_simple(model_name, max_length, use_bfloat16=True):
    """Simple model setup without FlashAttention"""
    
    success, modules = import_transformers_safely()
    if not success:
        raise ImportError("Failed to import transformers safely")
    
    (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, 
     DataCollatorForLanguageModeling, BitsAndBytesConfig, LoraConfig, 
     get_peft_model, TaskType, prepare_model_for_kbit_training, Dataset) = modules
    
    print(f"🤖 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization for A100
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
        attn_implementation="eager"  # Explicitly avoid FlashAttention
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer, (TrainingArguments, Trainer, DataCollatorForLanguageModeling, Dataset)

def create_training_dataset(training_data, tokenizer, max_length):
    """Create dataset for training"""
    _, _, _, Dataset = tokenizer, max_length, None, None  # Get Dataset from return
    
    # Simple dataset creation
    texts = []
    for example in training_data:
        # Create interview format
        text = f"Question: {example['input']}\nAnswer: {example['output']}"
        texts.append(text)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset

def main():
    """Main A100 training function"""
    print("🚀 A100 Simple Training - FlashAttention-Free")
    print("=" * 60)
    
    try:
        # Step 1: Check GPU
        a100_available = check_gpu()
        
        # Step 2: Create training data
        print("\n📚 Creating technical questions database...")
        from technical_questions_db import TechnicalQuestionsDatabase
        questions_db = TechnicalQuestionsDatabase()
        print(f"✅ Created database with {len(questions_db.questions)} questions")
        
        # Generate training data
        scenarios = 150 if a100_available else 100
        print(f"\n🎯 Generating {scenarios} training scenarios...")
        from enhanced_data_processor import TechnicalInterviewDataset as DatasetGenerator
        dataset_creator = DatasetGenerator(questions_db)
        training_data = dataset_creator.create_realistic_interview_scenarios(scenarios)
        print(f"✅ Generated {len(training_data)} training examples")
        
        # Step 3: Setup model
        print("\n🤖 Setting up model (FlashAttention-free)...")
        max_length = 1024 if a100_available else 512
        use_bfloat16 = a100_available
        
        model, tokenizer, training_modules = setup_model_simple(
            "codellama/CodeLlama-7b-Instruct-hf", 
            max_length, 
            use_bfloat16
        )
        
        TrainingArguments, Trainer, DataCollatorForLanguageModeling, _ = training_modules
        
        # Step 4: Create dataset
        print("\n📊 Creating training dataset...")
        train_dataset = create_training_dataset(training_data, tokenizer, max_length)
        
        # Create a small eval dataset
        eval_dataset = train_dataset.select(range(min(50, len(train_dataset))))
        
        # Step 5: Setup training
        print("\n🏋️ Setting up trainer...")
        
        batch_size = 4 if a100_available else 1
        learning_rate = 1e-5 if a100_available else 2e-5
        
        training_args = TrainingArguments(
            output_dir="./technical_interview_model",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 8 // batch_size),
            num_train_epochs=3,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=25,
            save_steps=100,
            eval_steps=100,
            evaluation_strategy="steps",
            save_total_limit=3,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            bf16=use_bfloat16,
            fp16=not use_bfloat16,
            report_to=None  # Disable wandb
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        print("✅ Trainer configured successfully")
        
        # Step 6: Train
        print(f"\n🏃 Starting A100 training...")
        start_time = datetime.now()
        
        Path("./technical_interview_model").mkdir(parents=True, exist_ok=True)
        
        trainer.train()
        
        training_time = datetime.now() - start_time
        print(f"\n🎉 Training completed in {training_time}!")
        
        # Step 7: Save model
        print("\n💾 Saving model...")
        trainer.save_model("./technical_interview_model")
        tokenizer.save_pretrained("./technical_interview_model")
        print("✅ Model saved to ./technical_interview_model")
        
        # Cleanup
        del trainer, model
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n🎯 A100 Training Complete!")
        print(f"⏱️ Training time: {training_time}")
        print(f"📊 Training examples: {len(training_data)}")
        if a100_available:
            print("⚡ A100 power delivered - 13x faster than T4!")
        print("🎯 Model ready for testing!")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 