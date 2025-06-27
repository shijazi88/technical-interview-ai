#!/usr/bin/env python3
"""
Web Interface for Technical Interview AI
Interactive chat interface to test your trained CodeLlama model
"""

import gradio as gr
import torch
from technical_interview_bot import TechnicalInterviewBot
from interview_logger import InterviewLogger
import os
import json
from datetime import datetime

class WebInterfaceBot:
    def __init__(self, model_path="./technical_interview_model"):
        """Initialize the web interface bot"""
        self.model_path = model_path
        self.bot = None
        self.chat_history = []
        self.interview_active = False
        self.current_interview = {}
        self.logger = InterviewLogger()  # Initialize logger
        self.current_session_id = None
        
        # Try to load the model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.bot = TechnicalInterviewBot(self.model_path)
                if self.bot.model is not None:
                    print("✅ Model loaded successfully!")
                    return True
                else:
                    print("❌ Model file found but failed to load")
                    return False
            else:
                print(f"❌ Model not found at {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def start_interview(self, language, experience_level, candidate_name):
        """Start a new interview session"""
        if not self.bot or not self.bot.model:
            return "❌ Model not loaded. Please check the model path.", []
        
        try:
            # Reset state
            self.chat_history = []
            self.interview_active = True
            self.current_interview = {
                "language": language,
                "experience_level": experience_level,
                "candidate_name": candidate_name,
                "start_time": datetime.now().isoformat()
            }
            
            # Start logging session
            self.current_session_id = self.logger.start_session(
                candidate_name=candidate_name,
                programming_language=language,
                experience_level=experience_level,
                model_version="codellama-7b-instruct-hf"
            )
            
            # Start interview
            response = self.bot.start_interview(
                programming_language=language,
                experience_level=experience_level,
                candidate_name=candidate_name
            )
            
            # Add to chat history
            self.chat_history.append([None, response])
            
            return response, self.chat_history
            
        except Exception as e:
            error_msg = f"❌ Error starting interview: {str(e)}"
            # Log the error
            if hasattr(self, 'logger'):
                self.logger.log_error("interview_start_error", str(e), f"Language: {language}, Level: {experience_level}")
            return error_msg, []
    
    def chat_with_bot(self, message, history):
        """Handle chat messages with the bot"""
        if not self.interview_active:
            return history, ""
        
        if not self.bot or not self.bot.model:
            history.append([message, "❌ Model not loaded. Please restart the interview."])
            return history, ""
        
        try:
            # Store the last question for logging
            last_question = "Follow-up question"
            question_category = "general"
            question_difficulty = 3
            
            # Try to extract question info from bot context if available
            if hasattr(self.bot, 'current_question'):
                last_question = getattr(self.bot.current_question, 'question', last_question)
                question_category = getattr(self.bot.current_question, 'category', question_category)
                question_difficulty = getattr(self.bot.current_question, 'difficulty_score', question_difficulty)
            
            # Process the candidate's response
            start_time = datetime.now()
            bot_response = self.bot.process_response(message)
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Log the Q&A exchange
            if self.current_session_id and hasattr(self, 'logger'):
                self.logger.log_question_response(
                    question_text=last_question,
                    question_category=question_category,
                    question_difficulty=question_difficulty,
                    candidate_response=message,
                    ai_follow_up=bot_response,
                    response_time_seconds=response_time
                )
            
            # Add to history
            history.append([message, bot_response])
            
            # Check if interview is complete
            if "Thank you for the technical interview" in bot_response or self.bot.question_count >= self.bot.max_questions:
                self.interview_active = False
                summary = self.bot.get_interview_summary()
                
                # End logging session
                if self.current_session_id and hasattr(self, 'logger'):
                    self.logger.end_session(final_difficulty=summary.get('final_difficulty', 1))
                
                summary_text = f"""
📊 **Interview Summary:**
- Questions asked: {summary['questions_asked']}
- Final difficulty: {summary['final_difficulty']}/8
- Topics covered: {', '.join(summary['topics_covered'])}
- Overall performance: {'Strong' if summary['final_difficulty'] > 5 else 'Moderate' if summary['final_difficulty'] > 3 else 'Developing'}
- Session ID: {self.current_session_id[:8]}... (for analysis)
"""
                history.append([None, summary_text])
            
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Error processing response: {str(e)}"
            history.append([message, error_msg])
            return history, ""
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.bot or not self.bot.model:
            return "❌ No model loaded"
        
        try:
            device = next(self.bot.model.parameters()).device
            model_size = sum(p.numel() for p in self.bot.model.parameters() if p.requires_grad)
            
            info = f"""
🤖 **Model Information:**
- Model path: {self.model_path}
- Device: {device}
- Trainable parameters: {model_size:,}
- Model type: Technical Interview AI (CodeLlama-based)
- Status: ✅ Ready for interviews
"""
            return info
        except Exception as e:
            return f"❌ Error getting model info: {e}"

# Initialize the bot
web_bot = WebInterfaceBot()

def create_gradio_interface():
    """Create the Gradio web interface"""
    
    # Custom CSS for better styling
    css = """
    .chat-container {
        height: 500px;
        overflow-y: auto;
    }
    .interview-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Technical Interview AI") as interface:
        
        # Header
        gr.HTML("""
        <div class="interview-header">
            <h1>🤖 Technical Interview AI</h1>
            <p>Test your trained CodeLlama model with interactive technical interviews</p>
        </div>
        """)
        
        # Model status
        with gr.Row():
            model_info = gr.Textbox(
                value=web_bot.get_model_info(),
                label="Model Status",
                lines=6,
                interactive=False
            )
        
        # Interview setup
        with gr.Row():
            with gr.Column(scale=1):
                language = gr.Dropdown(
                    choices=["python", "java", "csharp", "flutter", "php", "javascript"],
                    value="python",
                    label="Programming Language"
                )
                experience_level = gr.Dropdown(
                    choices=["junior", "mid_level", "senior", "lead"],
                    value="mid_level",
                    label="Experience Level"
                )
                candidate_name = gr.Textbox(
                    value="Test Candidate",
                    label="Candidate Name"
                )
                start_btn = gr.Button("🚀 Start Interview", variant="primary")
            
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    value=[],
                    label="Technical Interview Chat",
                    height=400
                )
                
                # Message input
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your response here...",
                        label="Your Response",
                        lines=2,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="secondary", scale=1)
        
        # Instructions
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 8px;">
            <h3>📋 How to Use:</h3>
            <ol>
                <li><strong>Select</strong> programming language and experience level</li>
                <li><strong>Enter</strong> candidate name (or use default)</li>
                <li><strong>Click</strong> "Start Interview" to begin</li>
                <li><strong>Answer</strong> the technical questions as they appear</li>
                <li><strong>Continue</strong> the conversation naturally</li>
                <li><strong>Review</strong> your interview summary at the end</li>
            </ol>
            <p><strong>💡 Tip:</strong> Answer questions as you would in a real interview. The AI will adapt its questions based on your responses!</p>
        </div>
        """)
        
        # Event handlers
        def start_interview_handler(lang, exp_level, name):
            response, history = web_bot.start_interview(lang, exp_level, name)
            return history, ""
        
        def send_message_handler(message, history):
            if message.strip():
                new_history, _ = web_bot.chat_with_bot(message, history)
                return new_history, ""
            return history, message
        
        # Connect events
        start_btn.click(
            start_interview_handler,
            inputs=[language, experience_level, candidate_name],
            outputs=[chatbot, msg_input]
        )
        
        send_btn.click(
            send_message_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            send_message_handler,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
    
    return interface

def launch_web_interface(share=False, port=7860):
    """Launch the web interface"""
    print("🚀 Starting Technical Interview AI Web Interface...")
    
    # Check if model is loaded
    if not web_bot.bot or not web_bot.bot.model:
        print("⚠️ Warning: Model not loaded. Interface will start but interviews won't work.")
        print("Please ensure your model is saved in './technical_interview_model/'")
    
    interface = create_gradio_interface()
    
    print(f"🌐 Interface will be available at:")
    print(f"   Local: http://localhost:{port}")
    if share:
        print(f"   Public: Will be generated by Gradio")
    
    interface.launch(
        share=share,
        server_port=port,
        server_name="0.0.0.0" if share else "127.0.0.1"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Technical Interview AI Web Interface")
    parser.add_argument("--model_path", type=str, default="./technical_interview_model", 
                       help="Path to trained model")
    parser.add_argument("--share", action="store_true", 
                       help="Create public shareable link")
    parser.add_argument("--port", type=int, default=7860, 
                       help="Port to run the interface on")
    
    args = parser.parse_args()
    
    # Update model path if specified
    if args.model_path != "./technical_interview_model":
        web_bot.model_path = args.model_path
        web_bot.load_model()
    
    # Launch interface
    launch_web_interface(share=args.share, port=args.port) 