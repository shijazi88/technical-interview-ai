from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

class ExperienceLevel(Enum):
    """Experience levels for technical interviews"""
    JUNIOR = "junior"           # 0-2 years
    MID_LEVEL = "mid_level"     # 2-5 years  
    SENIOR = "senior"           # 5+ years
    LEAD = "lead"               # 8+ years, leadership experience

class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVA = "java"
    CSHARP = "csharp"
    FLUTTER = "flutter"
    PHP = "php"
    JAVASCRIPT = "javascript"

class QuestionCategory(Enum):
    """Question categories for balanced interviews"""
    FUNDAMENTALS = "fundamentals"       # Basic language concepts
    OOP = "oop"                        # Object-oriented programming
    DESIGN_PATTERNS = "design_patterns" # Software design patterns
    FRAMEWORKS = "frameworks"          # Language-specific frameworks
    DEBUGGING = "debugging"            # Problem-solving skills
    ARCHITECTURE = "architecture"      # System design concepts

@dataclass
class TechnicalQuestion:
    """Technical interview question with metadata"""
    id: str
    question: str
    language: ProgrammingLanguage
    category: QuestionCategory
    experience_level: ExperienceLevel
    expected_keywords: List[str]  # For automated evaluation
    follow_up_hints: List[str]    # If candidate struggles
    sample_answer: str
    difficulty_score: int         # 1-10 scale

class TechnicalQuestionsDatabase:
    """Central repository for all technical questions"""
    
    def __init__(self):
        self.questions: List[TechnicalQuestion] = []
        self._load_default_questions()
    
    def _load_default_questions(self):
        """Load predefined technical questions"""
        
        # Python Questions
        python_questions = [
            TechnicalQuestion(
                id="py_001",
                question="What is the difference between a list and a tuple in Python?",
                language=ProgrammingLanguage.PYTHON,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["mutable", "immutable", "ordered", "list", "tuple"],
                follow_up_hints=[
                    "Think about whether you can modify them after creation",
                    "Consider performance differences",
                    "When would you use each one?"
                ],
                sample_answer="Lists are mutable (can be changed) and use square brackets []. Tuples are immutable (cannot be changed) and use parentheses (). Lists are better for data that changes, tuples for fixed data like coordinates.",
                difficulty_score=2
            ),
            
            TechnicalQuestion(
                id="py_002", 
                question="Explain dependency injection and how you would implement it in Python.",
                language=ProgrammingLanguage.PYTHON,
                category=QuestionCategory.DESIGN_PATTERNS,
                experience_level=ExperienceLevel.SENIOR,
                expected_keywords=["dependency", "injection", "constructor", "interface", "decoupling"],
                follow_up_hints=[
                    "Think about how objects get their dependencies",
                    "Consider testing benefits",
                    "What about using frameworks like FastAPI?"
                ],
                sample_answer="Dependency injection is providing dependencies to an object rather than having it create them. In Python, you can inject via constructor parameters, use protocols for interfaces, or use frameworks like dependency-injector.",
                difficulty_score=7
            ),
            
            TechnicalQuestion(
                id="py_003",
                question="What is a decorator in Python and can you write a simple caching decorator?",
                language=ProgrammingLanguage.PYTHON,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.MID_LEVEL,
                expected_keywords=["decorator", "wrapper", "function", "@", "caching"],
                follow_up_hints=[
                    "Think about the @ syntax",
                    "How do you preserve function metadata?",
                    "What about function arguments?"
                ],
                sample_answer="A decorator modifies or extends function behavior. A caching decorator: @functools.lru_cache or custom implementation storing results in a dict to avoid recomputation.",
                difficulty_score=5
            ),
            
            TechnicalQuestion(
                id="py_004",
                question="Explain the difference between __str__ and __repr__ in Python.",
                language=ProgrammingLanguage.PYTHON,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["__str__", "__repr__", "string representation", "debugging", "user-friendly"],
                follow_up_hints=[
                    "Think about when each method is called",
                    "Which one is for developers vs users?",
                    "What happens if only one is defined?"
                ],
                sample_answer="__str__ returns user-friendly string representation, __repr__ returns developer-friendly unambiguous representation. If __str__ is missing, __repr__ is used as fallback.",
                difficulty_score=3
            )
        ]
        
        # Java Questions
        java_questions = [
            TechnicalQuestion(
                id="java_001",
                question="What is the difference between an interface and an abstract class in Java?",
                language=ProgrammingLanguage.JAVA,
                category=QuestionCategory.OOP,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["interface", "abstract", "implements", "extends", "multiple inheritance"],
                follow_up_hints=[
                    "Think about multiple inheritance",
                    "Consider what can contain implementations",
                    "What about default methods in interfaces?"
                ],
                sample_answer="Interfaces define contracts (what must be implemented) and support multiple inheritance. Abstract classes can have partial implementations and fields, but only single inheritance. Java 8+ interfaces can have default methods.",
                difficulty_score=3
            ),
            
            TechnicalQuestion(
                id="java_002",
                question="Explain the Spring dependency injection container and different types of injection.",
                language=ProgrammingLanguage.JAVA,
                category=QuestionCategory.FRAMEWORKS,
                experience_level=ExperienceLevel.SENIOR,
                expected_keywords=["Spring", "IOC", "constructor", "setter", "field", "autowired"],
                follow_up_hints=[
                    "What are the different ways to inject dependencies?",
                    "Which injection type is recommended and why?",
                    "How does Spring resolve circular dependencies?"
                ],
                sample_answer="Spring IOC container manages object lifecycle. Three injection types: constructor (recommended for required deps), setter (optional deps), and field injection (not recommended). Constructor injection ensures immutability and prevents circular dependencies.",
                difficulty_score=8
            ),
            
            TechnicalQuestion(
                id="java_003",
                question="What is the difference between == and .equals() in Java?",
                language=ProgrammingLanguage.JAVA,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["==", "equals", "reference", "value", "object comparison"],
                follow_up_hints=[
                    "Think about primitive vs object types",
                    "What does == compare for objects?",
                    "When should you override equals()?"
                ],
                sample_answer="== compares references (memory addresses) for objects, values for primitives. .equals() compares object content/values. Always use .equals() for string comparison, override equals() for custom objects.",
                difficulty_score=2
            )
        ]
        
        # C# Questions
        csharp_questions = [
            TechnicalQuestion(
                id="cs_001",
                question="What is the difference between value types and reference types in C#?",
                language=ProgrammingLanguage.CSHARP,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["value type", "reference type", "stack", "heap", "struct", "class"],
                follow_up_hints=[
                    "Think about memory allocation",
                    "What happens when you assign one to another?",
                    "Examples of each type?"
                ],
                sample_answer="Value types (int, struct) are stored on stack, copied by value. Reference types (class, string) are stored on heap, copied by reference. Value types are independent copies, reference types share the same object.",
                difficulty_score=4
            ),
            
            TechnicalQuestion(
                id="cs_002",
                question="Explain async/await pattern in C# and how it differs from Task.Run().",
                language=ProgrammingLanguage.CSHARP,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.MID_LEVEL,
                expected_keywords=["async", "await", "Task", "thread", "asynchronous", "IO-bound"],
                follow_up_hints=[
                    "When should you use async/await vs Task.Run?",
                    "What happens to the calling thread?",
                    "What about ConfigureAwait(false)?"
                ],
                sample_answer="async/await is for I/O-bound operations, doesn't block calling thread. Task.Run() is for CPU-bound work, uses thread pool. async/await returns to original context, Task.Run() switches threads.",
                difficulty_score=6
            )
        ]
        
        # Flutter/Dart Questions
        flutter_questions = [
            TechnicalQuestion(
                id="flutter_001",
                question="What is the difference between StatefulWidget and StatelessWidget in Flutter?",
                language=ProgrammingLanguage.FLUTTER,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["StatefulWidget", "StatelessWidget", "state", "rebuild", "setState"],
                follow_up_hints=[
                    "When does the widget rebuild?",
                    "What triggers a rebuild?",
                    "Performance considerations?"
                ],
                sample_answer="StatelessWidget is immutable, rebuilds when parent changes. StatefulWidget has mutable state, can trigger rebuilds via setState(). Use StatelessWidget for static content, StatefulWidget for dynamic UI.",
                difficulty_score=3
            ),
            
            TechnicalQuestion(
                id="flutter_002", 
                question="Explain Flutter's widget tree and how the rendering pipeline works.",
                language=ProgrammingLanguage.FLUTTER,
                category=QuestionCategory.ARCHITECTURE,
                experience_level=ExperienceLevel.SENIOR,
                expected_keywords=["widget tree", "element tree", "render tree", "build", "layout", "paint"],
                follow_up_hints=[
                    "What are the three trees in Flutter?",
                    "How does Flutter optimize rebuilds?",
                    "What happens during the layout phase?"
                ],
                sample_answer="Flutter has three trees: Widget (configuration), Element (lifecycle), and RenderObject (layout/paint). Build phase creates widgets, layout calculates sizes, paint draws to screen. Keys help optimize rebuilds.",
                difficulty_score=8
            )
        ]
        
        # PHP Questions
        php_questions = [
            TechnicalQuestion(
                id="php_001",
                question="What is the difference between include, require, include_once, and require_once in PHP?",
                language=ProgrammingLanguage.PHP,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["include", "require", "once", "warning", "fatal error"],
                follow_up_hints=[
                    "What happens when the file is not found?",
                    "When would you use _once versions?",
                    "Performance considerations?"
                ],
                sample_answer="include/require load files. include gives warning if file missing, require gives fatal error. _once versions prevent duplicate inclusion. Use require for critical files, include for optional ones.",
                difficulty_score=2
            ),
            
            TechnicalQuestion(
                id="php_002",
                question="Explain PSR-4 autoloading and how modern PHP dependency injection works.",
                language=ProgrammingLanguage.PHP,
                category=QuestionCategory.DESIGN_PATTERNS,
                experience_level=ExperienceLevel.SENIOR,
                expected_keywords=["PSR-4", "autoloading", "namespace", "composer", "dependency injection", "container"],
                follow_up_hints=[
                    "How does composer autoloading work?",
                    "What are PSR standards?",
                    "How do DI containers resolve dependencies?"
                ],
                sample_answer="PSR-4 maps namespaces to directory structure for autoloading. Composer generates autoloader based on composer.json. DI containers like Symfony's resolve dependencies automatically using reflection and configuration.",
                difficulty_score=7
            )
        ]
        
        # JavaScript Questions
        javascript_questions = [
            TechnicalQuestion(
                id="js_001",
                question="Explain the difference between var, let, and const in JavaScript.",
                language=ProgrammingLanguage.JAVASCRIPT,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.JUNIOR,
                expected_keywords=["var", "let", "const", "scope", "hoisting", "block scope"],
                follow_up_hints=[
                    "Think about scope differences",
                    "What is hoisting?",
                    "When would you use each one?"
                ],
                sample_answer="var has function scope and is hoisted. let and const have block scope. const cannot be reassigned. let is for variables that change, const for constants and objects that won't be reassigned.",
                difficulty_score=3
            ),
            
            TechnicalQuestion(
                id="js_002",
                question="What are closures in JavaScript and how do they work?",
                language=ProgrammingLanguage.JAVASCRIPT,
                category=QuestionCategory.FUNDAMENTALS,
                experience_level=ExperienceLevel.MID_LEVEL,
                expected_keywords=["closure", "lexical scope", "inner function", "outer function", "memory"],
                follow_up_hints=[
                    "Think about scope and lifetime",
                    "How do closures access outer variables?",
                    "What are practical uses of closures?"
                ],
                sample_answer="Closures give inner functions access to outer function's variables even after outer function returns. The inner function 'closes over' variables from outer scope. Used for data privacy, callbacks, and module patterns.",
                difficulty_score=5
            )
        ]
        
        # Combine all questions
        self.questions.extend(python_questions)
        self.questions.extend(java_questions)
        self.questions.extend(csharp_questions)
        self.questions.extend(flutter_questions)
        self.questions.extend(php_questions)
        self.questions.extend(javascript_questions)
    
    def get_questions_by_criteria(self, 
                                 language: Optional[ProgrammingLanguage] = None,
                                 experience_level: Optional[ExperienceLevel] = None,
                                 category: Optional[QuestionCategory] = None,
                                 max_difficulty: Optional[int] = None) -> List[TechnicalQuestion]:
        """Filter questions based on criteria"""
        filtered = self.questions
        
        if language:
            filtered = [q for q in filtered if q.language == language]
        
        if experience_level:
            filtered = [q for q in filtered if q.experience_level == experience_level]
            
        if category:
            filtered = [q for q in filtered if q.category == category]
            
        if max_difficulty:
            filtered = [q for q in filtered if q.difficulty_score <= max_difficulty]
            
        return filtered
    
    def export_to_training_format(self, output_file: str):
        """Export questions in format suitable for model training"""
        training_data = []
        
        for question in self.questions:
            # Create instruction-following format
            instruction = f"""You are a technical interviewer conducting a {question.language.value} interview for a {question.experience_level.value} level position. 
Ask follow-up questions based on the candidate's response to assess their technical knowledge."""
            
            # Simulate candidate giving a basic answer
            input_context = f"""Question asked: {question.question}
Candidate's response: {question.sample_answer}
Interview focus: {question.category.value}
Experience level: {question.experience_level.value}"""
            
            # Generate appropriate follow-up
            follow_up = self._generate_follow_up(question)
            
            training_data.append({
                "instruction": instruction,
                "input": input_context,
                "output": follow_up,
                "metadata": {
                    "language": question.language.value,
                    "category": question.category.value,
                    "experience_level": question.experience_level.value,
                    "difficulty": question.difficulty_score
                }
            })
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"Exported {len(training_data)} training samples to {output_file}")
    
    def _generate_follow_up(self, question: TechnicalQuestion) -> str:
        """Generate appropriate follow-up questions"""
        follow_ups = {
            ExperienceLevel.JUNIOR: [
                "Can you give me a practical example of when you would use this?",
                "What do you think are the main benefits of this approach?",
                "Have you encountered any challenges with this in your projects?"
            ],
            ExperienceLevel.MID_LEVEL: [
                "How would you explain this concept to a junior developer?",
                "What are some alternative approaches, and when would you choose each?",
                "Can you walk me through how you would implement this in a real project?"
            ],
            ExperienceLevel.SENIOR: [
                "What are the performance implications of this approach?",
                "How would you design a system that scales this concept to handle millions of users?",
                "What are some common pitfalls developers face with this, and how do you avoid them?"
            ]
        }
        
        import random
        level_questions = follow_ups.get(question.experience_level, follow_ups[ExperienceLevel.MID_LEVEL])
        return random.choice(level_questions)

# Usage Example
if __name__ == "__main__":
    # Initialize database
    db = TechnicalQuestionsDatabase()
    
    # Example: Get Python questions for junior developers
    python_junior_questions = db.get_questions_by_criteria(
        language=ProgrammingLanguage.PYTHON,
        experience_level=ExperienceLevel.JUNIOR
    )
    
    print(f"Found {len(python_junior_questions)} Python junior questions:")
    for q in python_junior_questions:
        print(f"- {q.question}")
    
    # Export for training
    db.export_to_training_format("technical_interview_dataset.json") 