from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from torch.utils.data import Dataset
import json
from typing import Dict, List
import gc
import os

class TechnicalInterviewTokenizer:
    """Specialized tokenizer setup for technical interviews"""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-7b-Instruct-hf"):
        self.model_name = model_name
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._add_technical_tokens()
    
    def _add_technical_tokens(self):
        """Add specialized tokens for technical interviews"""
        
        special_tokens = {
            "additional_special_tokens": [
                # Interview role markers
                "<|interviewer|>", "<|candidate|>", 
                
                # Programming languages
                "<|python|>", "<|java|>", "<|csharp|>", "<|flutter|>", "<|php|>", "<|javascript|>",
                
                # Experience levels
                "<|junior|>", "<|mid_level|>", "<|senior|>", "<|lead|>",
                
                # Question categories
                "<|fundamentals|>", "<|oop|>", "<|design_patterns|>", 
                "<|frameworks|>", "<|debugging|>", "<|architecture|>",
                
                # Conversation flow
                "<|question|>", "<|response|>", "<|follow_up|>", "<|context|>",
                
                # Quality indicators
                "<|struggling|>", "<|confident|>", "<|expert_level|>"
            ]
        }
        
        # Add the special tokens
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens for technical interviews")
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_interview_prompt(self, 
                               question: str,
                               candidate_response: str, 
                               language: str,
                               experience_level: str,
                               category: str,
                               context: str = "") -> str:
        """Create standardized prompt format for technical interviews"""
        
        prompt = f"""<|{language}|><|{experience_level}|><|{category}|>
<|context|>{context}

<|interviewer|><|question|>
{question}

<|candidate|><|response|>
{candidate_response}

<|interviewer|><|follow_up|>"""
        
        return prompt

class TechnicalInterviewDataset(Dataset):
    """PyTorch Dataset for technical interview training"""
    
    def __init__(self, data_path: str, tokenizer: TechnicalInterviewTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load training data
        print(f"Loading training data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} training examples")
        
        # Preprocess data for efficiency
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess data to extract key information"""
        valid_examples = []
        
        for example in self.data:
            try:
                # Validate required fields
                required_fields = ['instruction', 'input', 'output', 'metadata']
                if all(field in example for field in required_fields):
                    valid_examples.append(example)
                else:
                    print(f"Skipping invalid example: missing fields")
            except Exception as e:
                print(f"Error processing example: {e}")
        
        self.data = valid_examples
        print(f"Preprocessed {len(self.data)} valid examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get training example at index"""
        example = self.data[idx]
        metadata = example['metadata']
        
        # Extract context from input
        input_parts = example['input'].split('\n')
        question = ""
        candidate_response = ""
        
        for part in input_parts:
            if part.startswith("Previous Question:"):
                question = part.replace("Previous Question:", "").strip()
            elif part.startswith("Candidate's Response:"):
                candidate_response = part.replace("Candidate's Response:", "").strip()
        
        # Create full conversation prompt
        prompt = self.tokenizer.create_interview_prompt(
            question=question,
            candidate_response=candidate_response,
            language=metadata['language'],
            experience_level=metadata['experience_level'],
            category=metadata['category'],
            context=f"Turn {metadata.get('conversation_turn', 1)} of technical interview"
        )
        
        # Add the target output
        full_text = prompt + example['output'] + self.tokenizer.tokenizer.eos_token
        
        # Tokenize
        encoding = self.tokenizer.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # For causal LM, labels = input_ids
        }

class TechnicalInterviewLoRA:
    """LoRA configuration optimized for technical interviews"""
    
    def __init__(self, base_model, target_modules=None):
        self.base_model = base_model
        
        # Default target modules for conversation models
        self.target_modules = target_modules or [
            "c_attn",  # For GPT-style models (attention)
            "c_proj",  # For GPT-style models (projection)  
            "c_fc",    # For GPT-style models (feed-forward)
        ]
    
    def create_config(self, rank: int = 16, alpha: int = 32, dropout: float = 0.1):
        """Create LoRA configuration for technical interviews (optimized for Colab)"""
        
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,                    # Reduced rank for memory efficiency
            lora_alpha=alpha,          # 2x rank is common practice
            lora_dropout=dropout,      # LoRA dropout
            target_modules=self.target_modules,
            bias="none",               # Don't adapt bias terms
            inference_mode=False,      # Training mode
        )
        
        return config
    
    def apply_lora(self, config):
        """Apply LoRA to the base model"""
        
        # Apply LoRA
        model = get_peft_model(self.base_model, config)
        
        # Print parameter information
        model.print_trainable_parameters()
        
        return model

class TechnicalInterviewTrainer:
    """Specialized trainer for technical interview models"""
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
    
    def create_training_arguments(self, output_dir: str = "./technical_interview_model"):
        """Create training arguments optimized for Google Colab Pro"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # Training schedule
            num_train_epochs=3,                    # Reduced for faster training
            per_device_train_batch_size=1,         # Small batch for memory
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,         # Effective batch size = 4
            
            # Learning rate and optimization
            learning_rate=2e-5,                    # Conservative for fine-tuning
            warmup_steps=100,                      # Gradual warmup
            weight_decay=0.01,                     # Regularization
            
            # Efficiency settings
            fp16=True,                             # Mixed precision
            dataloader_pin_memory=False,           # Reduce memory usage
            gradient_checkpointing=True,           # Trade compute for memory
            
            # Logging and evaluation
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            
            # Model management
            save_total_limit=2,                    # Keep only best models
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Reproducibility
            seed=42,
            data_seed=42,
            
            # Other settings
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for Colab
            
            # Memory optimization
            dataloader_num_workers=0,              # Single process for Colab
            max_grad_norm=1.0,                     # Gradient clipping
        )
        
        return training_args
    
    def create_trainer(self, training_args):
        """Create Hugging Face trainer with custom settings"""
        
        # Data collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer.tokenizer,
            data_collator=data_collator,
        )
        
        return trainer

def setup_technical_interview_training(
    num_scenarios: int = 100, 
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
    max_length: int = 512
):
    """
    Complete setup for technical interview model training
    Optimized for Google Colab Pro
    
    Args:
        num_scenarios: Number of training scenarios to generate
        model_name: Base model to use (default: "codellama/CodeLlama-7b-Instruct-hf")
        max_length: Maximum sequence length for training (default: 512 for 7B models)
    """
    
    print("🚀 Setting up technical interview training for Google Colab Pro...")
    print(f"📊 Base model: {model_name}")
    print(f"🎯 Training scenarios: {num_scenarios}")
    print(f"📏 Max length: {max_length}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 1. Generate training data first
    print("\n📚 Step 1: Generating training data...")
    from technical_questions_db import TechnicalQuestionsDatabase
    from enhanced_data_processor import TechnicalInterviewDataset as DatasetGenerator
    
    questions_db = TechnicalQuestionsDatabase()
    dataset_creator = DatasetGenerator(questions_db)
    training_data = dataset_creator.create_realistic_interview_scenarios(num_scenarios)
    
    # Save training data
    training_file = "technical_interview_training_data.json"
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Generated {len(training_data)} training examples")
    
    # 2. Load base model
    print(f"\n🤖 Step 2: Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # More efficient loading for large models
        use_cache=False          # Disable cache during training to save memory
    )
    
    # 3. Setup tokenizer
    print("\n🔤 Step 3: Setting up specialized tokenizer...")
    tech_tokenizer = TechnicalInterviewTokenizer(model_name)
    
    # Resize model embeddings to accommodate new tokens
    base_model.resize_token_embeddings(len(tech_tokenizer.tokenizer))
    
    # 4. Create datasets
    print("\n📊 Step 4: Creating datasets...")
    train_dataset = TechnicalInterviewDataset(
        training_file, 
        tech_tokenizer,
        max_length=max_length
    )
    
    # Create smaller eval dataset for faster evaluation
    eval_data = training_data[:min(50, len(training_data)//10)]
    eval_file = "eval_data.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    eval_dataset = TechnicalInterviewDataset(
        eval_file, 
        tech_tokenizer,
        max_length=max_length
    )
    
    # 5. Apply LoRA
    print("\n⚡ Step 5: Applying LoRA...")
    
    # Determine target modules based on model architecture
    if "codellama" in model_name.lower() or "llama" in model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "mistral" in model_name.lower():
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:  # DialoGPT and other GPT-style models
        target_modules = ["c_attn", "c_proj", "c_fc"]
    
    lora_setup = TechnicalInterviewLoRA(base_model, target_modules=target_modules)
    lora_config = lora_setup.create_config(rank=16, alpha=32, dropout=0.1)  # Optimized for Colab
    model = lora_setup.apply_lora(lora_config)
    
    # 6. Setup trainer
    print("\n🏋️ Step 6: Setting up trainer...")
    trainer_setup = TechnicalInterviewTrainer(
        model, tech_tokenizer, train_dataset, eval_dataset
    )
    
    training_args = trainer_setup.create_training_arguments()
    trainer = trainer_setup.create_trainer(training_args)
    
    print("✅ Setup complete! Ready to start training.")
    
    # Memory check
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    
    return trainer, tech_tokenizer

# Run setup
if __name__ == "__main__":
    trainer, tokenizer = setup_technical_interview_training(num_scenarios=100)
    
    print("\n🚀 Starting training...")
    trainer.train()
    
    print("\n💾 Saving model...")
    trainer.save_model("./final_technical_interview_model")
    tokenizer.tokenizer.save_pretrained("./final_technical_interview_model")
    
    print("✅ Training completed!") 