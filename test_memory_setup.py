#!/usr/bin/env python3
"""
Memory-optimized setup test for Technical Interview AI
Tests model loading and setup without full training to verify memory fixes
"""

import torch
import gc
import psutil
import os

def get_memory_info():
    """Get current memory usage information"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(0) / 1e9
        gpu_reserved = torch.cuda.memory_reserved(0) / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_free = gpu_total - gpu_reserved
    else:
        gpu_memory = gpu_reserved = gpu_total = gpu_free = 0
    
    ram_usage = psutil.virtual_memory()
    
    return {
        'gpu_allocated': gpu_memory,
        'gpu_reserved': gpu_reserved, 
        'gpu_total': gpu_total,
        'gpu_free': gpu_free,
        'ram_used': ram_usage.used / 1e9,
        'ram_total': ram_usage.total / 1e9,
        'ram_percent': ram_usage.percent
    }

def test_memory_optimized_setup():
    """Test the memory-optimized model setup"""
    print("🧪 Testing Memory-Optimized Setup")
    print("=" * 50)
    
    # Initial memory check
    print("📊 Initial Memory State:")
    mem_info = get_memory_info()
    print(f"  GPU: {mem_info['gpu_allocated']:.1f}GB allocated, {mem_info['gpu_free']:.1f}GB free")
    print(f"  RAM: {mem_info['ram_used']:.1f}GB used ({mem_info['ram_percent']:.1f}%)")
    
    try:
        # Test data generation (should be super fast now)
        print(f"\n🎯 Step 1: Testing data generation...")
        start_time = time.time()
        
        from technical_questions_db import TechnicalQuestionsDatabase
        from enhanced_data_processor import TechnicalInterviewDataset as DatasetGenerator
        
        questions_db = TechnicalQuestionsDatabase()
        dataset_creator = DatasetGenerator(questions_db)
        
        # Test with small number for speed
        training_data = dataset_creator.create_realistic_interview_scenarios(5)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(training_data)} examples in {generation_time:.2f}s")
        
        # Memory check after data generation
        mem_info = get_memory_info()
        print(f"📊 After data generation:")
        print(f"  GPU: {mem_info['gpu_allocated']:.1f}GB allocated")
        print(f"  RAM: {mem_info['ram_used']:.1f}GB used ({mem_info['ram_percent']:.1f}%)")
        
        # Test model setup
        print(f"\n🤖 Step 2: Testing model setup...")
        from technical_model_setup import setup_technical_interview_training
        
        # This should work with memory optimizations
        trainer, tokenizer = setup_technical_interview_training(
            num_scenarios=5,  # Very small for testing
            model_name="codellama/CodeLlama-7b-Instruct-hf",
            max_length=256    # Smaller for memory efficiency
        )
        
        # Memory check after model loading
        mem_info = get_memory_info()
        print(f"📊 After model setup:")
        print(f"  GPU: {mem_info['gpu_allocated']:.1f}GB allocated, {mem_info['gpu_free']:.1f}GB free")
        print(f"  RAM: {mem_info['ram_used']:.1f}GB used ({mem_info['ram_percent']:.1f}%)")
        
        # Test a quick training step (without full training)
        print(f"\n🏃 Step 3: Testing training preparation...")
        
        # Just verify trainer is ready
        train_dataloader = trainer.get_train_dataloader()
        print(f"✅ Training dataloader ready with {len(train_dataloader)} batches")
        
        # Get one batch to test
        batch = next(iter(train_dataloader))
        print(f"✅ Batch shape: {batch['input_ids'].shape}")
        
        # Memory check after dataloader
        mem_info = get_memory_info()
        print(f"📊 After training prep:")
        print(f"  GPU: {mem_info['gpu_allocated']:.1f}GB allocated, {mem_info['gpu_free']:.1f}GB free")
        print(f"  RAM: {mem_info['ram_used']:.1f}GB used ({mem_info['ram_percent']:.1f}%)")
        
        print(f"\n🎉 Memory Test PASSED!")
        print(f"✅ Model setup working with {mem_info['gpu_free']:.1f}GB GPU memory remaining")
        print(f"✅ Ready for actual training!")
        
        # Cleanup
        del trainer, tokenizer, batch, train_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Memory Test FAILED: {e}")
        
        # Final memory check
        mem_info = get_memory_info()
        print(f"📊 Final Memory State:")
        print(f"  GPU: {mem_info['gpu_allocated']:.1f}GB allocated, {mem_info['gpu_free']:.1f}GB free")
        print(f"  RAM: {mem_info['ram_used']:.1f}GB used ({mem_info['ram_percent']:.1f}%)")
        
        print(f"\n🔧 Memory optimization needed!")
        print(f"💡 Try reducing batch size or using smaller model")
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return False

if __name__ == "__main__":
    import time
    success = test_memory_optimized_setup()
    
    if success:
        print(f"\n🚀 Ready to run full training:")
        print(f"   python colab_training_pipeline.py --num_scenarios 20")
    else:
        print(f"\n⚠️ Fix memory issues before attempting full training") 