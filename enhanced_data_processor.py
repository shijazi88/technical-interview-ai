from typing import List, Dict, Optional, Tuple
import json
import random
from dataclasses import asdict
from technical_questions_db import (
    TechnicalQuestionsDatabase, 
    ProgrammingLanguage, 
    ExperienceLevel, 
    QuestionCategory,
    TechnicalQuestion
)

class InterviewContextManager:
    """Manages interview context and conversation flow"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.topics_covered: set = set()
        self.current_difficulty: int = 1
        self.candidate_skill_indicators: Dict[str, int] = {}
    
    def add_exchange(self, question: TechnicalQuestion, candidate_response: str, 
                    interviewer_follow_up: str):
        """Add question-response exchange to history"""
        self.conversation_history.append({
            "question": question.question,
            "question_id": question.id,
            "category": question.category.value,
            "difficulty": question.difficulty_score,
            "candidate_response": candidate_response,
            "interviewer_follow_up": interviewer_follow_up,
            "timestamp": len(self.conversation_history)
        })
        
        # Track topics
        self.topics_covered.add(question.category.value)
        
        # Assess response quality (simplified)
        response_quality = self._assess_response_quality(question, candidate_response)
        self.candidate_skill_indicators[question.category.value] = response_quality
        
        # Adjust difficulty
        self._adjust_difficulty(response_quality)
    
    def _assess_response_quality(self, question: TechnicalQuestion, response: str) -> int:
        """Simple keyword-based response assessment"""
        keywords_found = 0
        response_lower = response.lower()
        
        for keyword in question.expected_keywords:
            if keyword.lower() in response_lower:
                keywords_found += 1
        
        # Score from 1-5 based on keyword coverage
        if len(question.expected_keywords) == 0:
            return 3  # Default score
        
        coverage = keywords_found / len(question.expected_keywords)
        if coverage >= 0.8:
            return 5  # Excellent
        elif coverage >= 0.6:
            return 4  # Good
        elif coverage >= 0.4:
            return 3  # Average
        elif coverage >= 0.2:
            return 2  # Below average
        else:
            return 1  # Poor
    
    def _adjust_difficulty(self, response_quality: int):
        """Adjust interview difficulty based on candidate performance"""
        if response_quality >= 4:
            self.current_difficulty = min(10, self.current_difficulty + 1)
        elif response_quality <= 2:
            self.current_difficulty = max(1, self.current_difficulty - 1)
        # No change for average responses (3)
    
    def get_context_summary(self) -> str:
        """Generate summary of interview progress"""
        if not self.conversation_history:
            return "Interview just started."
        
        topics = ", ".join(self.topics_covered)
        avg_performance = sum(self.candidate_skill_indicators.values()) / len(self.candidate_skill_indicators) if self.candidate_skill_indicators else 3.0
        
        return f"""Interview Progress:
- Questions asked: {len(self.conversation_history)}
- Topics covered: {topics}
- Current difficulty level: {self.current_difficulty}
- Average performance: {avg_performance:.1f}/5.0
- Strongest area: {max(self.candidate_skill_indicators.items(), key=lambda x: x[1])[0] if self.candidate_skill_indicators else 'None'}"""

class TechnicalInterviewDataset:
    """Enhanced dataset creation with realistic interview flows"""
    
    def __init__(self, questions_db: TechnicalQuestionsDatabase):
        self.questions_db = questions_db
        self.context_manager = InterviewContextManager()
    
    def create_realistic_interview_scenarios(self, num_scenarios: int = 100) -> List[Dict]:
        """Create complete interview scenarios with multiple exchanges"""
        scenarios = []
        
        print(f"🎯 Generating {num_scenarios} realistic interview scenarios...")
        print("📊 Progress tracking enabled - you'll see time per scenario")
        print("=" * 60)
        
        import time
        start_time = time.time()
        scenario_times = []
        
        for i in range(num_scenarios):
            scenario_start = time.time()
            
            # Show immediate feedback
            print(f"🔄 Starting scenario {i + 1}/{num_scenarios}...", end=" ", flush=True)
            
            # Generate the scenario
            scenario = self._generate_single_interview()
            scenarios.extend(scenario)
            
            # Show completion
            print(f"✅ Done ({len(scenario)} examples)", flush=True)
            
            # Calculate timing
            scenario_time = time.time() - scenario_start
            scenario_times.append(scenario_time)
            
            # Progress reporting
            progress_pct = ((i + 1) / num_scenarios) * 100
            elapsed_total = time.time() - start_time
            
            # Estimate remaining time
            if len(scenario_times) >= 3:
                avg_time_per_scenario = sum(scenario_times[-5:]) / len(scenario_times[-5:])  # Use last 5 for better estimate
                remaining_scenarios = num_scenarios - (i + 1)
                estimated_remaining = remaining_scenarios * avg_time_per_scenario
                remaining_str = self._format_time(estimated_remaining)
                eta_str = self._format_time(elapsed_total + estimated_remaining)
            else:
                remaining_str = "Calculating..."
                eta_str = "Calculating..."
            
            # Show progress every scenario (first 10) then every 5 scenarios
            if i < 10 or (i + 1) % 5 == 0 or (i + 1) == num_scenarios:
                examples_in_scenario = len(scenario)
                print(f"✅ Scenario {i + 1:3d}/{num_scenarios} ({progress_pct:5.1f}%) | "
                      f"⏱️ {scenario_time:4.1f}s | "
                      f"📝 {examples_in_scenario} examples | "
                      f"⏳ Remaining: {remaining_str} | "
                      f"🎯 Total ETA: {eta_str}")
        
        total_time = time.time() - start_time
        avg_scenario_time = sum(scenario_times) / len(scenario_times)
        
        print("=" * 60)
        print(f"🎉 Data Generation Complete!")
        print(f"📊 Generated {len(scenarios)} total training examples from {num_scenarios} scenarios")
        print(f"⏱️ Total time: {self._format_time(total_time)}")
        print(f"📈 Average per scenario: {avg_scenario_time:.1f}s")
        print(f"🚀 Ready for model training!")
        
        return scenarios
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def _generate_single_interview(self) -> List[Dict]:
        """Generate a complete interview with 5-8 questions"""
        # Reset context for new interview
        self.context_manager = InterviewContextManager()
        
        # Choose random language and experience level
        language = random.choice(list(ProgrammingLanguage))
        experience = random.choice(list(ExperienceLevel))
        
        interview_exchanges = []
        questions_asked = 0
        max_questions = random.randint(5, 8)
        
        # Start with fundamentals
        current_category = QuestionCategory.FUNDAMENTALS
        categories_tried = set()
        no_questions_count = 0  # Safety counter to prevent infinite loops
        
        while questions_asked < max_questions:
            # Safety check to prevent infinite loops
            if no_questions_count > 20:  # If we've tried 20 times without finding questions
                print(f"⚠️ Warning: Could not find enough questions for {language.value} {experience.value}")
                break
                
            # Get appropriate questions with gradually relaxed criteria
            candidate_questions = self._get_questions_with_fallback(
                language, experience, current_category, no_questions_count
            )
            
            if not candidate_questions:
                no_questions_count += 1
                categories_tried.add(current_category)
                
                # Try different category if no questions found
                remaining_categories = set(QuestionCategory) - categories_tried
                if remaining_categories:
                    current_category = random.choice(list(remaining_categories))
                else:
                    # Reset and try any category again with more relaxed criteria
                    categories_tried.clear()
                    current_category = random.choice(list(QuestionCategory))
                continue
            
            # Reset counter since we found questions
            no_questions_count = 0
            
            # Select question
            question = random.choice(candidate_questions)
            
            # Generate realistic candidate response
            candidate_response = self._generate_candidate_response(question, experience)
            
            # Generate interviewer follow-up
            interviewer_follow_up = self._generate_interviewer_follow_up(
                question, candidate_response, experience
            )
            
            # Create training example
            training_example = self._create_training_example(
                question, candidate_response, interviewer_follow_up, 
                language, experience
            )
            
            interview_exchanges.append(training_example)
            
            # Update context
            self.context_manager.add_exchange(question, candidate_response, interviewer_follow_up)
            
            questions_asked += 1
            
            # Progress to more advanced topics
            if questions_asked == 2:
                current_category = QuestionCategory.OOP
            elif questions_asked == 4:
                current_category = random.choice([
                    QuestionCategory.DESIGN_PATTERNS,
                    QuestionCategory.FRAMEWORKS,
                    QuestionCategory.ARCHITECTURE
                ])
        
        return interview_exchanges
    
    def _get_questions_with_fallback(self, language: ProgrammingLanguage, 
                                   experience: ExperienceLevel, 
                                   category: QuestionCategory,
                                   attempt_number: int) -> List[TechnicalQuestion]:
        """Get questions with progressively relaxed criteria to prevent infinite loops"""
        
        # First few attempts: strict criteria
        if attempt_number < 5:
            return self.questions_db.get_questions_by_criteria(
                language=language,
                experience_level=experience,
                category=category,
                max_difficulty=self.context_manager.current_difficulty + 2
            )
        
        # Next attempts: relax difficulty constraint
        elif attempt_number < 10:
            return self.questions_db.get_questions_by_criteria(
                language=language,
                experience_level=experience,
                category=category
                # No max_difficulty constraint
            )
        
        # Next attempts: relax experience level
        elif attempt_number < 15:
            return self.questions_db.get_questions_by_criteria(
                language=language,
                category=category
                # No experience_level constraint
            )
        
        # Final attempts: only match language
        else:
            return self.questions_db.get_questions_by_criteria(
                language=language
                # Only language constraint
            )
    
    def _generate_candidate_response(self, question: TechnicalQuestion, 
                                   experience: ExperienceLevel) -> str:
        """Generate realistic candidate responses with varying quality"""
        
        # Response quality based on experience and question difficulty
        base_quality = {
            ExperienceLevel.JUNIOR: 2.5,
            ExperienceLevel.MID_LEVEL: 3.5,
            ExperienceLevel.SENIOR: 4.0,
            ExperienceLevel.LEAD: 4.5
        }[experience]
        
        # Adjust for question difficulty
        difficulty_factor = question.difficulty_score / 10.0
        expected_quality = base_quality * (1 - difficulty_factor * 0.3)
        
        # Add randomness
        actual_quality = max(1, min(5, expected_quality + random.uniform(-1, 1)))
        
        if actual_quality >= 4:
            # Good response - use sample answer with minor variations
            variations = [
                question.sample_answer,
                f"Well, {question.sample_answer.lower()}",
                f"From my experience, {question.sample_answer.lower()}",
                f"I think {question.sample_answer.lower()}"
            ]
            return random.choice(variations)
        
        elif actual_quality >= 3:
            # Average response - partial knowledge
            keywords = question.expected_keywords[:len(question.expected_keywords)//2]
            return f"I know that {' and '.join(keywords[:2])} are important, but I'm not completely sure about all the details."
        
        elif actual_quality >= 2:
            # Below average - minimal knowledge
            if question.expected_keywords:
                return f"I think it has something to do with {question.expected_keywords[0]}, but I'm not entirely confident."
            else:
                return "I'm not completely sure about this one. Could you give me a hint?"
        
        else:
            # Poor response - honest admission
            return "I haven't worked with this much. Could you explain it or give me some guidance?"
    
    def _generate_interviewer_follow_up(self, question: TechnicalQuestion,
                                      candidate_response: str, 
                                      experience: ExperienceLevel) -> str:
        """Generate appropriate interviewer follow-up based on response quality"""
        
        response_lower = candidate_response.lower()
        
        # Detect if candidate struggled
        struggle_indicators = [
            "not sure", "don't know", "not familiar", "haven't worked",
            "could you explain", "give me a hint", "not confident"
        ]
        
        is_struggling = any(indicator in response_lower for indicator in struggle_indicators)
        
        if is_struggling:
            # Provide guidance or easier follow-up
            if question.follow_up_hints:
                return f"No problem! {random.choice(question.follow_up_hints)} Let me give you a hint to help you think about it."
            else:
                return "That's okay! Let me ask you something related but perhaps easier to start with."
        
        else:
            # Good response - dig deeper
            advanced_follow_ups = {
                ExperienceLevel.JUNIOR: [
                    "Great! Can you think of a real-world example where you might use this?",
                    "That's a good start. What do you think might be some challenges with this approach?",
                    "Excellent! How do you think this compares to alternative approaches?"
                ],
                ExperienceLevel.MID_LEVEL: [
                    "Good explanation! How would you implement this in a production environment?",
                    "That's correct. What are some performance considerations you'd keep in mind?",
                    "Nice! Can you walk me through how you'd test this functionality?"
                ],
                ExperienceLevel.SENIOR: [
                    "Excellent! How would you architect this to handle enterprise-scale requirements?",
                    "Perfect! What are some common anti-patterns you've seen with this approach?",
                    "Great depth! How would you mentor a junior developer learning this concept?"
                ],
                ExperienceLevel.LEAD: [
                    "Outstanding! How would you evangelize this approach across multiple teams?",
                    "Impressive! What governance policies would you put in place around this?",
                    "Excellent leadership perspective! How do you balance technical debt with this approach?"
                ]
            }
            
            follow_ups = advanced_follow_ups.get(experience, advanced_follow_ups[ExperienceLevel.MID_LEVEL])
            return random.choice(follow_ups)
    
    def _create_training_example(self, question: TechnicalQuestion,
                               candidate_response: str, interviewer_follow_up: str,
                               language: ProgrammingLanguage, 
                               experience: ExperienceLevel) -> Dict:
        """Create properly formatted training example"""
        
        instruction = f"""You are conducting a technical interview for a {experience.value} {language.value} developer position. 
Based on the candidate's response to your question, provide an appropriate follow-up that:
1. Assesses their technical depth
2. Maintains a supportive interview atmosphere
3. Explores their practical experience
4. Adapts to their demonstrated knowledge level"""

        input_context = f"""Previous Question: {question.question}
Question Category: {question.category.value}
Question Difficulty: {question.difficulty_score}/10
Candidate's Response: {candidate_response}
Experience Level: {experience.value}
Programming Language: {language.value}
Interview Context: {self.context_manager.get_context_summary()}"""

        return {
            "instruction": instruction,
            "input": input_context,
            "output": interviewer_follow_up,
            "metadata": {
                "question_id": question.id,
                "language": language.value,
                "experience_level": experience.value,
                "category": question.category.value,
                "difficulty": question.difficulty_score,
                "conversation_turn": len(self.context_manager.conversation_history)
            }
        }

# Usage Example
if __name__ == "__main__":
    # Create questions database
    questions_db = TechnicalQuestionsDatabase()
    
    # Create enhanced dataset
    dataset_creator = TechnicalInterviewDataset(questions_db)
    training_scenarios = dataset_creator.create_realistic_interview_scenarios(50)
    
    # Save training data
    with open('technical_interview_training_data.json', 'w', encoding='utf-8') as f:
        json.dump(training_scenarios, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(training_scenarios)} training examples from realistic interview scenarios") 