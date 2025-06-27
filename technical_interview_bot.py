from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import List, Dict, Optional
import random
import os

class TechnicalInterviewBot:
    """Production-ready technical interview bot optimized for Google Colab"""
    
    def __init__(self, model_path: str = "./final_technical_interview_model"):
        self.model_path = model_path
        self.max_tokens = os.getenv('MAX_TOKENS', '1000')  # Using memory setting
        
        # Initialize with smaller settings for Colab
        self.max_tokens = min(1000, int(self.max_tokens))
        
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Model not found at {model_path}. Please train the model first.")
            self.model = None
            self.tokenizer = None
            
        self.reset_interview()
    
    def load_model(self):
        """Load the trained technical interview model"""
        print("Loading technical interview model...")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Please ensure the model has been trained and saved properly.")
            self.model = None
            self.tokenizer = None
    
    def reset_interview(self):
        """Reset interview state for new candidate"""
        self.conversation_history = []
        self.topics_covered = set()
        self.current_difficulty = 3  # Start at medium difficulty
        self.candidate_performance = {}
        self.question_count = 0
        self.max_questions = 8
    
    def start_interview(self, 
                       programming_language: str,
                       experience_level: str, 
                       candidate_name: str = "Candidate",
                       job_title: str = "Software Engineer") -> str:
        """Start a new technical interview session"""
        
        if self.model is None:
            return "❌ Model not loaded. Please train the model first using the training pipeline."
        
        self.reset_interview()
        self.programming_language = programming_language.lower()
        self.experience_level = experience_level.lower()
        self.candidate_name = candidate_name
        self.job_title = job_title
        
        # Welcome message
        welcome = f"""Hello {candidate_name}! I'm your AI technical interviewer for the {job_title} position.

I'll be focusing on {programming_language.title()} questions appropriate for a {experience_level.replace('_', ' ')} level developer. 

The interview will be conversational - I'll ask follow-up questions based on your responses to better understand your experience and knowledge depth.

Let's begin with a foundational question."""
        
        # Generate first question
        first_question = self._get_opening_question()
        
        full_response = f"{welcome}\n\n**Question 1:** {first_question}"
        
        # Track the question
        self.conversation_history.append({
            "type": "question",
            "content": first_question,
            "question_number": 1,
            "difficulty": self.current_difficulty
        })
        self.question_count = 1
        
        return full_response
    
    def _get_opening_question(self) -> str:
        """Get appropriate opening question based on language and experience"""
        
        opening_questions = {
            "python": {
                "junior": "Can you explain the difference between a list and a tuple in Python, and when you would use each one?",
                "mid_level": "What are Python decorators and can you give me an example of when you've used them?",
                "senior": "How would you handle memory management and performance optimization in a Python application?",
                "lead": "How do you approach Python code reviews and what standards do you enforce across your team?"
            },
            "java": {
                "junior": "What's the difference between an interface and an abstract class in Java?",
                "mid_level": "Can you explain the concept of dependency injection and how it's used in Spring?",
                "senior": "How would you design a thread-safe singleton pattern in Java?",
                "lead": "What strategies do you use for managing technical debt in large Java codebases?"
            },
            "csharp": {
                "junior": "What's the difference between value types and reference types in C#?",
                "mid_level": "Can you explain async/await in C# and when you would use it?",
                "senior": "How would you implement a custom middleware in ASP.NET Core?",
                "lead": "What's your approach to .NET architecture decisions for enterprise applications?"
            },
            "flutter": {
                "junior": "What's the difference between StatefulWidget and StatelessWidget in Flutter?",
                "mid_level": "How do you manage state in a Flutter application?",
                "senior": "How would you optimize Flutter app performance for both iOS and Android?",
                "lead": "What's your strategy for Flutter app architecture in large teams?"
            },
            "php": {
                "junior": "What's the difference between include and require in PHP?",
                "mid_level": "Can you explain PSR standards and why they're important?",
                "senior": "How would you implement a secure authentication system in PHP?",
                "lead": "What's your approach to PHP application architecture and framework selection?"
            },
            "javascript": {
                "junior": "Can you explain the difference between var, let, and const in JavaScript?",
                "mid_level": "What are closures in JavaScript and how do they work?",
                "senior": "How would you optimize JavaScript performance in a large-scale application?",
                "lead": "What's your approach to JavaScript architecture and team coding standards?"
            }
        }
        
        language_questions = opening_questions.get(self.programming_language, opening_questions["python"])
        return language_questions.get(self.experience_level, language_questions["mid_level"])
    
    def process_response(self, candidate_response: str) -> str:
        """Process candidate's response and generate follow-up question"""
        
        if not candidate_response.strip():
            return "I didn't receive your response. Could you please share your thoughts on the previous question?"
        
        if self.model is None:
            return "❌ Model not loaded. Please train the model first."
        
        # Add response to history
        self.conversation_history.append({
            "type": "response", 
            "content": candidate_response,
            "question_number": self.question_count
        })
        
        # Assess response quality
        response_quality = self._assess_response_quality(candidate_response)
        
        # Update candidate performance tracking
        current_topic = self._get_current_topic()
        self.candidate_performance[current_topic] = response_quality
        
        # Adjust difficulty based on performance
        self._adjust_difficulty(response_quality)
        
        # Check if interview should continue
        if self.question_count >= self.max_questions:
            return self._conclude_interview()
        
        # Generate follow-up question
        follow_up = self._generate_follow_up_question(candidate_response, response_quality)
        
        self.question_count += 1
        self.conversation_history.append({
            "type": "question",
            "content": follow_up,
            "question_number": self.question_count,
            "difficulty": self.current_difficulty
        })
        
        return f"**Question {self.question_count}:** {follow_up}"
    
    def _assess_response_quality(self, response: str) -> int:
        """Assess the quality of candidate's response (1-5 scale)"""
        
        response_lower = response.lower()
        
        # Check for confidence/uncertainty indicators
        uncertainty_phrases = [
            "not sure", "don't know", "i think", "maybe", "probably",
            "not familiar", "haven't used", "could be wrong"
        ]
        
        confidence_phrases = [
            "definitely", "in my experience", "i've used", "i would",
            "the best practice", "typically", "always", "never"
        ]
        
        # Check for technical depth
        technical_indicators = [
            "because", "however", "therefore", "for example", "such as",
            "implementation", "performance", "security", "scalability"
        ]
        
        # Calculate base score
        score = 3  # Start with average
        
        # Length factor
        if len(response.split()) < 10:
            score -= 1
        elif len(response.split()) > 50:
            score += 1
        
        # Confidence factor
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
        confidence_count = sum(1 for phrase in confidence_phrases if phrase in response_lower)
        
        if uncertainty_count > confidence_count:
            score -= 1
        elif confidence_count > uncertainty_count:
            score += 1
        
        # Technical depth factor
        technical_count = sum(1 for indicator in technical_indicators if indicator in response_lower)
        if technical_count >= 2:
            score += 1
        elif technical_count == 0:
            score -= 1
        
        return max(1, min(5, score))
    
    def _adjust_difficulty(self, response_quality: int):
        """Adjust interview difficulty based on candidate performance"""
        if response_quality >= 4:
            self.current_difficulty = min(8, self.current_difficulty + 1)
        elif response_quality <= 2:
            self.current_difficulty = max(2, self.current_difficulty - 1)
    
    def _get_current_topic(self) -> str:
        """Determine current topic based on question number"""
        topics = ["fundamentals", "oop", "frameworks", "architecture", "best_practices"]
        topic_index = min(self.question_count - 1, len(topics) - 1)
        return topics[topic_index]
    
    def _generate_follow_up_question(self, candidate_response: str, quality: int) -> str:
        """Generate contextual follow-up question using the trained model"""
        
        # Get the last question asked
        last_question = ""
        for entry in reversed(self.conversation_history):
            if entry["type"] == "question":
                last_question = entry["content"]
                break
        
        # Create prompt for the model
        prompt = f"""<|{self.programming_language}|><|{self.experience_level}|><|{self._get_current_topic()}|>
<|context|>Turn {self.question_count} of technical interview

<|interviewer|><|question|>
{last_question}

<|candidate|><|response|>
{candidate_response}

<|interviewer|><|follow_up|>"""
        
        try:
            # Generate response using the model
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Limit input length to avoid memory issues
            max_input_length = min(self.max_tokens // 2, 512)
            if inputs.shape[1] > max_input_length:
                inputs = inputs[:, -max_input_length:]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the follow-up question
            follow_up = generated_text.split("<|follow_up|>")[-1].strip()
            
            # Clean up the response
            follow_up = follow_up.split("<|")[0].strip()  # Remove any trailing special tokens
            
            # Fallback to rule-based if generation fails
            if not follow_up or len(follow_up) < 10:
                follow_up = self._fallback_question(quality)
            
            return follow_up
            
        except Exception as e:
            print(f"Error generating follow-up: {e}")
            return self._fallback_question(quality)
    
    def _fallback_question(self, quality: int) -> str:
        """Fallback questions if model generation fails"""
        
        if quality <= 2:
            # Struggling candidate - provide help
            encouraging_questions = [
                "Let me approach this differently. Can you think of a time when you solved a similar problem?",
                "That's okay! Let's try a related but simpler example. Can you tell me about...",
                "No worries! Let me give you a hint to help you think through this."
            ]
            return random.choice(encouraging_questions)
        
        else:
            # Good response - dig deeper
            deeper_questions = [
                "That's a great explanation! Can you walk me through how you would implement this in a real project?",
                "Excellent! What are some potential challenges or edge cases you'd consider?",
                "Perfect! How would you explain this concept to a junior developer on your team?"
            ]
            return random.choice(deeper_questions)
    
    def _conclude_interview(self) -> str:
        """Generate interview conclusion with summary"""
        
        # Calculate overall performance
        if self.candidate_performance:
            avg_performance = sum(self.candidate_performance.values()) / len(self.candidate_performance)
            performance_level = "excellent" if avg_performance >= 4 else \
                              "good" if avg_performance >= 3 else \
                              "developing"
        else:
            performance_level = "engaged"
        
        conclusion = f"""Thank you for the technical interview, {self.candidate_name}! 

**Interview Summary:**
- Questions covered: {self.question_count}
- Topics discussed: {', '.join(self.topics_covered) if self.topics_covered else 'Various technical concepts'}
- Final difficulty level: {self.current_difficulty}/8
- Overall engagement: {performance_level.title()}

You demonstrated {performance_level} technical knowledge throughout our conversation. The hiring team will review your responses and follow up with next steps.

Is there anything specific about the role or our technical stack that you'd like to know more about?"""
        
        return conclusion
    
    def get_interview_summary(self) -> Dict:
        """Get structured summary of the interview for review"""
        
        return {
            "candidate_name": self.candidate_name,
            "job_title": self.job_title,
            "programming_language": self.programming_language,
            "experience_level": self.experience_level,
            "questions_asked": self.question_count,
            "topics_covered": list(self.topics_covered),
            "final_difficulty": self.current_difficulty,
            "performance_by_topic": self.candidate_performance,
            "conversation_history": self.conversation_history,
            "overall_assessment": "Needs detailed human review"
        }

# Demo function for testing
def demo_interview():
    """Demonstrate the technical interview bot in action"""
    
    print("=== Technical Interview Bot Demo ===\n")
    
    # Initialize bot
    bot = TechnicalInterviewBot()
    
    if bot.model is None:
        print("❌ Model not loaded. Please run the training pipeline first.")
        return
    
    # Start interview
    welcome = bot.start_interview(
        programming_language="python",
        experience_level="mid_level",
        candidate_name="Alex",
        job_title="Senior Python Developer"
    )
    
    print(welcome)
    print("\n" + "="*50 + "\n")
    
    # Simulate candidate responses
    sample_responses = [
        "Lists are mutable and use square brackets, while tuples are immutable and use parentheses. I use lists when I need to modify the data and tuples for fixed data like coordinates.",
        
        "Decorators are functions that modify other functions. I've used @property for getters/setters and @functools.lru_cache for memoization in a recent project where I needed to optimize expensive calculations.",
        
        "I would implement it using constructor injection with type hints. For example, I'd define interfaces using protocols and inject dependencies through the __init__ method. This makes testing easier since I can mock dependencies.",
        
        "For database connections, I use connection pooling with SQLAlchemy and implement retry logic with exponential backoff. I also make sure to use context managers for proper resource cleanup."
    ]
    
    # Process each response
    for i, response in enumerate(sample_responses, 1):
        print(f"**Candidate Response {i}:** {response}")
        print("\n" + "-"*30 + "\n")
        
        follow_up = bot.process_response(response)
        print(follow_up)
        print("\n" + "="*50 + "\n")
        
        if "Thank you for the technical interview" in follow_up:
            break
    
    # Show interview summary
    summary = bot.get_interview_summary()
    print("**Interview Summary:**")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    demo_interview() 