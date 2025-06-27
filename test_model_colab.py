# TEST YOUR TRAINED MODEL - Add this cell after training completes

# Test the model immediately after training
from technical_interview_bot import TechnicalInterviewBot
import os

# Check if model was saved successfully
model_path = "./technical_interview_model"
if os.path.exists(model_path):
    print(f"✅ Model found at: {model_path}")
    
    # List all files in the model directory
    print("\n📁 Model files:")
    for file in os.listdir(model_path):
        file_path = os.path.join(model_path, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file}: {size_mb:.1f} MB")
    
    print("\n🧪 Testing the trained model...")
    
    # Initialize the bot with your trained model
    bot = TechnicalInterviewBot(model_path)
    
    if bot.model is not None:
        print("✅ Model loaded successfully!")
        
        # Test with a Python interview
        print("\n🔥 Starting test interview...")
        response = bot.start_interview(
            programming_language="python",
            experience_level="mid_level", 
            candidate_name="CodeLlama Test"
        )
        
        print("🤖 AI Interviewer Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        # Test a follow-up response
        print("\n💬 Testing follow-up response...")
        follow_up = bot.process_response(
            "Lists are mutable and can be modified after creation, while tuples are immutable and cannot be changed."
        )
        
        print("🤖 AI Interviewer Follow-up:")
        print("-" * 50)
        print(follow_up)
        print("-" * 50)
        
        # Show interview summary
        summary = bot.get_interview_summary()
        print("\n📊 Interview Summary:")
        print(f"  - Questions asked: {summary['questions_asked']}")
        print(f"  - Topics covered: {', '.join(summary['topics_covered'])}")
        print(f"  - Final difficulty: {summary['final_difficulty']}/8")
        
        print("\n🎉 SUCCESS! Your CodeLlama model is working perfectly!")
        print("🌐 Ready to launch web interface for interactive testing!")
        
    else:
        print("❌ Model files found but failed to load.")
        print("Check the error messages above.")
        
else:
    print(f"❌ Model not found at: {model_path}")
    print("Make sure training completed successfully.")

# Optional: Show GPU memory usage
import torch
if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated(0) / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used") 