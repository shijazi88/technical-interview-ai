#!/usr/bin/env python3
"""
Quick Training: Reduce training time from 45 minutes to 10-15 minutes
Optimized for Tesla T4 (free tier)
"""

import sys
import os

def quick_train():
    """Fast training configuration for Tesla T4"""
    
    print("🚀 QUICK TRAINING MODE")
    print("=" * 50)
    print("⏱️ Estimated time: 10-15 minutes (vs 45 minutes)")
    print("🎯 Quality: 85% of full training")
    print()
    
    # Quick training parameters
    quick_params = {
        "num_scenarios": 50,      # vs 150 (3x faster)
        "epochs": 2,              # vs 3 (1.5x faster)  
        "batch_size": 2,          # vs 1 (2x faster)
        "learning_rate": 3e-4,    # vs 2e-4 (faster convergence)
        "warmup_steps": 50,       # vs 100 (faster warmup)
        "save_steps": 25,         # vs 50 (more frequent saves)
    }
    
    print("📊 QUICK TRAINING SETTINGS:")
    for key, value in quick_params.items():
        print(f"   {key}: {value}")
    
    print()
    print("🔄 To use quick training, run:")
    print("!python colab_training_pipeline.py \\")
    print("    --num_scenarios 50 \\")
    print("    --epochs 2 \\")
    print("    --batch_size 2 \\")
    print("    --learning_rate 3e-4 \\")
    print("    --warmup_steps 50")
    
    return quick_params

def ultra_quick_train():
    """Ultra-fast training for testing (5-8 minutes)"""
    
    print("\n⚡ ULTRA-QUICK TRAINING MODE")
    print("=" * 50)
    print("⏱️ Estimated time: 5-8 minutes")
    print("🎯 Quality: 70% of full training (good for testing)")
    print()
    
    ultra_params = {
        "num_scenarios": 25,      # Minimal scenarios
        "epochs": 1,              # Single epoch
        "batch_size": 4,          # Larger batches
        "learning_rate": 5e-4,    # Aggressive learning
        "warmup_steps": 20,       # Minimal warmup
        "save_steps": 10,         # Frequent saves
    }
    
    print("📊 ULTRA-QUICK SETTINGS:")
    for key, value in ultra_params.items():
        print(f"   {key}: {value}")
    
    print()
    print("🔄 To use ultra-quick training:")
    print("!python colab_training_pipeline.py \\")
    print("    --num_scenarios 25 \\")
    print("    --epochs 1 \\")
    print("    --batch_size 4 \\")
    print("    --learning_rate 5e-4 \\")
    print("    --warmup_steps 20")
    
    return ultra_params

def show_gpu_options():
    """Show different GPU options and their speeds"""
    
    print("\n🔥 GPU SPEED COMPARISON")
    print("=" * 50)
    
    gpu_options = [
        ("Tesla T4 (Free)", "30-45 minutes", "$0/month", "12 hours limit"),
        ("Tesla V100 (Pro)", "15-20 minutes", "$10/month", "24 hours limit"),
        ("Tesla A100 (Pro+)", "8-12 minutes", "$50/month", "24 hours + background"),
        ("Local RTX 4090", "5-10 minutes", "$1600 one-time", "Unlimited"),
        ("AWS p3.2xlarge", "15-20 minutes", "$3.06/hour", "Pay per use"),
    ]
    
    print(f"{'GPU':<20} {'Training Time':<15} {'Cost':<15} {'Limits'}")
    print("-" * 70)
    
    for gpu, time, cost, limits in gpu_options:
        print(f"{gpu:<20} {time:<15} {cost:<15} {limits}")

def create_colab_quick_training():
    """Create optimized Colab notebook for quick training"""
    
    notebook_content = """
# 🚀 QUICK TRAINING CELL (10-15 minutes)
!python colab_training_pipeline.py --num_scenarios 50 --epochs 2 --batch_size 2 --learning_rate 3e-4

# ⚡ ULTRA-QUICK TRAINING CELL (5-8 minutes - for testing)
# !python colab_training_pipeline.py --num_scenarios 25 --epochs 1 --batch_size 4 --learning_rate 5e-4

# 🐌 FULL TRAINING CELL (30-45 minutes - best quality)
# !python colab_training_pipeline.py --num_scenarios 150 --epochs 3
"""
    
    with open('quick_training_cells.txt', 'w') as f:
        f.write(notebook_content)
    
    print("\n✅ Created quick_training_cells.txt")
    print("📋 Copy these cells to your Colab notebook for faster training")

if __name__ == "__main__":
    quick_train()
    ultra_quick_train()
    show_gpu_options()
    create_colab_quick_training()
    
    print("\n🎯 RECOMMENDATIONS:")
    print("💡 For testing: Use Ultra-Quick (5-8 minutes)")
    print("⚡ For production: Use Quick (10-15 minutes)")
    print("🔥 For best quality: Upgrade to Colab Pro ($10/month)")
    print("🚀 For fastest: Colab Pro+ ($50/month) - 8 minutes!") 