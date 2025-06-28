#!/usr/bin/env python3
"""
Test if Hugging Face key is needed for your current setup
Run this to check if you can use your trained model without authentication
"""

import os
import sys

def test_model_access():
    """Test if we can access the trained model without HF key"""
    print("🧪 Testing Hugging Face Access...")
    print("=" * 50)
    
    # Check if HF token is already set
    hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
    if hf_token:
        print(f"✅ HF Token found: {hf_token[:10]}...")
    else:
        print("ℹ️ No HF Token found - testing without authentication")
    
    # Test 1: Can we import required libraries?
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers library accessible")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: Can we load the tokenizer for CodeLlama?
    try:
        print("\n🔤 Testing tokenizer access...")
        tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
        print("✅ CodeLlama tokenizer loaded successfully")
    except Exception as e:
        print(f"⚠️ Tokenizer access issue: {e}")
        if "rate limit" in str(e).lower() or "forbidden" in str(e).lower():
            print("💡 This might require a Hugging Face token")
            return False
    
    # Test 3: Check if trained model exists locally
    model_path = "./technical_interview_model"
    if os.path.exists(model_path):
        print(f"✅ Local trained model found at: {model_path}")
        
        # List model files
        try:
            files = os.listdir(model_path)
            print(f"📁 Model files: {len(files)} files found")
            for file in files[:5]:  # Show first 5
                print(f"   - {file}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")
        except Exception as e:
            print(f"⚠️ Error listing model files: {e}")
    else:
        print(f"⚠️ Local model not found at: {model_path}")
        print("💡 Your model was trained on Colab, not on this local machine")
    
    # Test 4: Try loading local model (if exists)
    if os.path.exists(model_path):
        try:
            print("\n🤖 Testing local model loading...")
            from technical_interview_bot import TechnicalInterviewBot
            bot = TechnicalInterviewBot(model_path)
            if bot.model is not None:
                print("✅ Local trained model loads successfully!")
                print("🎉 You can use your model without HF token!")
                return True
            else:
                print("⚠️ Model files exist but failed to load")
        except Exception as e:
            print(f"⚠️ Local model test failed: {e}")
    
    return True

def main():
    """Main test function"""
    print("🔍 Hugging Face Authentication Test")
    print("This will check if you need a HF token for your setup")
    print()
    
    success = test_model_access()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ RESULT: You likely DON'T need a Hugging Face token!")
        print("🎯 Your trained model should work without authentication")
        print("\n💡 If you encounter issues later, you can always add a token")
    else:
        print("⚠️ RESULT: You might need a Hugging Face token")
        print("🔑 Follow the instructions above to get a free token")
    
    print("\n🚀 Next steps:")
    if os.path.exists("./technical_interview_model"):
        print("1. Your model is already trained and ready!")
        print("2. Run: python web_interface.py --share")
    else:
        print("1. Your model was trained on Colab - test it there first")
        print("2. Or download the model from Colab to use locally")

if __name__ == "__main__":
    main() 