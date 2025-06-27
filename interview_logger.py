#!/usr/bin/env python3
"""
Interview Logger and Analysis System
Stores all interview sessions for quality analysis and model improvement
"""

import json
import sqlite3
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class InterviewSession:
    """Complete interview session data"""
    session_id: str
    candidate_name: str
    programming_language: str
    experience_level: str
    start_time: str
    end_time: Optional[str] = None
    questions_asked: int = 0
    final_difficulty: int = 1
    topics_covered: List[str] = None
    total_duration_minutes: Optional[float] = None
    model_version: str = "codellama-7b-instruct"
    status: str = "active"  # active, completed, error
    
    def __post_init__(self):
        if self.topics_covered is None:
            self.topics_covered = []

@dataclass  
class QuestionResponse:
    """Individual question-response exchange"""
    session_id: str
    question_id: str
    question_text: str
    question_category: str
    question_difficulty: int
    candidate_response: str
    ai_follow_up: str
    response_timestamp: str
    response_quality_score: Optional[int] = None
    response_time_seconds: Optional[float] = None
    ai_confidence_score: Optional[float] = None
    
@dataclass
class QualityMetrics:
    """Quality assessment for responses"""
    session_id: str
    question_id: str
    relevance_score: int  # 1-5
    technical_accuracy: int  # 1-5  
    communication_clarity: int  # 1-5
    depth_of_knowledge: int  # 1-5
    overall_rating: int  # 1-5
    reviewer_notes: str = ""
    review_timestamp: str = ""

class InterviewLogger:
    """Comprehensive interview logging and storage system"""
    
    def __init__(self, db_path: str = "./interview_logs.db"):
        self.db_path = db_path
        self.current_session: Optional[InterviewSession] = None
        self.session_start_time: Optional[datetime] = None
        
        # Create database and tables
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interview_sessions (
                session_id TEXT PRIMARY KEY,
                candidate_name TEXT,
                programming_language TEXT,
                experience_level TEXT,
                start_time TEXT,
                end_time TEXT,
                questions_asked INTEGER,
                final_difficulty INTEGER,
                topics_covered TEXT,
                total_duration_minutes REAL,
                model_version TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Questions and responses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS question_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question_id TEXT,
                question_text TEXT,
                question_category TEXT,
                question_difficulty INTEGER,
                candidate_response TEXT,
                ai_follow_up TEXT,
                response_timestamp TEXT,
                response_quality_score INTEGER,
                response_time_seconds REAL,
                ai_confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES interview_sessions (session_id)
            )
        ''')
        
        # Quality metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question_id TEXT,
                relevance_score INTEGER,
                technical_accuracy INTEGER,
                communication_clarity INTEGER,
                depth_of_knowledge INTEGER,
                overall_rating INTEGER,
                reviewer_notes TEXT,
                review_timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES interview_sessions (session_id)
            )
        ''')
        
        # Error logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                error_type TEXT,
                error_message TEXT,
                error_timestamp TEXT,
                context_info TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def start_session(self, candidate_name: str, programming_language: str, 
                     experience_level: str, model_version: str = "codellama-7b-instruct") -> str:
        """Start a new interview session"""
        session_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc).isoformat()
        
        self.current_session = InterviewSession(
            session_id=session_id,
            candidate_name=candidate_name,
            programming_language=programming_language,
            experience_level=experience_level,
            start_time=start_time,
            model_version=model_version
        )
        
        self.session_start_time = datetime.now(timezone.utc)
        
        # Save to database
        self._save_session()
        
        print(f"📝 Started logging session: {session_id}")
        return session_id
    
    def log_question_response(self, question_text: str, question_category: str,
                            question_difficulty: int, candidate_response: str,
                            ai_follow_up: str, response_time_seconds: Optional[float] = None):
        """Log a question-response exchange"""
        if not self.current_session:
            print("⚠️ No active session. Call start_session() first.")
            return
        
        question_id = str(uuid.uuid4())
        response_timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create response record
        response = QuestionResponse(
            session_id=self.current_session.session_id,
            question_id=question_id,
            question_text=question_text,
            question_category=question_category,
            question_difficulty=question_difficulty,
            candidate_response=candidate_response,
            ai_follow_up=ai_follow_up,
            response_timestamp=response_timestamp,
            response_time_seconds=response_time_seconds
        )
        
        # Save to database
        self._save_question_response(response)
        
        # Update session stats
        self.current_session.questions_asked += 1
        if question_category not in self.current_session.topics_covered:
            self.current_session.topics_covered.append(question_category)
        
        print(f"📝 Logged Q&A exchange (Question #{self.current_session.questions_asked})")
    
    def end_session(self, final_difficulty: int = 1):
        """End the current interview session"""
        if not self.current_session:
            print("⚠️ No active session to end.")
            return
        
        end_time = datetime.now(timezone.utc)
        self.current_session.end_time = end_time.isoformat()
        self.current_session.final_difficulty = final_difficulty
        self.current_session.status = "completed"
        
        # Calculate duration
        if self.session_start_time:
            duration = (end_time - self.session_start_time).total_seconds() / 60
            self.current_session.total_duration_minutes = round(duration, 2)
        
        # Save final session state
        self._save_session()
        
        print(f"✅ Session completed: {self.current_session.session_id}")
        print(f"📊 Questions asked: {self.current_session.questions_asked}")
        print(f"⏱️ Duration: {self.current_session.total_duration_minutes} minutes")
        
        session_id = self.current_session.session_id
        self.current_session = None
        self.session_start_time = None
        
        return session_id
    
    def log_error(self, error_type: str, error_message: str, context_info: str = ""):
        """Log an error during interview"""
        session_id = self.current_session.session_id if self.current_session else "unknown"
        error_timestamp = datetime.now(timezone.utc).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO error_logs (session_id, error_type, error_message, error_timestamp, context_info)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, error_type, error_message, error_timestamp, context_info))
        
        conn.commit()
        conn.close()
        
        print(f"❌ Logged error: {error_type}")
    
    def add_quality_review(self, session_id: str, question_id: str, 
                          relevance: int, accuracy: int, clarity: int, 
                          depth: int, overall: int, notes: str = ""):
        """Add manual quality review for a response"""
        review_timestamp = datetime.now(timezone.utc).isoformat()
        
        metrics = QualityMetrics(
            session_id=session_id,
            question_id=question_id,
            relevance_score=relevance,
            technical_accuracy=accuracy,
            communication_clarity=clarity,
            depth_of_knowledge=depth,
            overall_rating=overall,
            reviewer_notes=notes,
            review_timestamp=review_timestamp
        )
        
        self._save_quality_metrics(metrics)
        print(f"📊 Added quality review for question {question_id}")
    
    def _save_session(self):
        """Save session to database"""
        if not self.current_session:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO interview_sessions 
            (session_id, candidate_name, programming_language, experience_level,
             start_time, end_time, questions_asked, final_difficulty, topics_covered,
             total_duration_minutes, model_version, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_session.session_id,
            self.current_session.candidate_name,
            self.current_session.programming_language,
            self.current_session.experience_level,
            self.current_session.start_time,
            self.current_session.end_time,
            self.current_session.questions_asked,
            self.current_session.final_difficulty,
            json.dumps(self.current_session.topics_covered),
            self.current_session.total_duration_minutes,
            self.current_session.model_version,
            self.current_session.status
        ))
        
        conn.commit()
        conn.close()
    
    def _save_question_response(self, response: QuestionResponse):
        """Save question response to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO question_responses 
            (session_id, question_id, question_text, question_category, question_difficulty,
             candidate_response, ai_follow_up, response_timestamp, response_quality_score,
             response_time_seconds, ai_confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            response.session_id, response.question_id, response.question_text,
            response.question_category, response.question_difficulty,
            response.candidate_response, response.ai_follow_up,
            response.response_timestamp, response.response_quality_score,
            response.response_time_seconds, response.ai_confidence_score
        ))
        
        conn.commit()
        conn.close()
    
    def _save_quality_metrics(self, metrics: QualityMetrics):
        """Save quality metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO quality_metrics 
            (session_id, question_id, relevance_score, technical_accuracy,
             communication_clarity, depth_of_knowledge, overall_rating,
             reviewer_notes, review_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.session_id, metrics.question_id, metrics.relevance_score,
            metrics.technical_accuracy, metrics.communication_clarity,
            metrics.depth_of_knowledge, metrics.overall_rating,
            metrics.reviewer_notes, metrics.review_timestamp
        ))
        
        conn.commit()
        conn.close()

class InterviewAnalyzer:
    """Analyze stored interview data for insights and improvements"""
    
    def __init__(self, db_path: str = "./interview_logs.db"):
        self.db_path = db_path
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get complete summary of a specific session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('SELECT * FROM interview_sessions WHERE session_id = ?', (session_id,))
        session_row = cursor.fetchone()
        
        if not session_row:
            return {"error": "Session not found"}
        
        # Get all questions and responses
        cursor.execute('''
            SELECT question_text, question_category, question_difficulty,
                   candidate_response, ai_follow_up, response_timestamp
            FROM question_responses 
            WHERE session_id = ? 
            ORDER BY created_at
        ''', (session_id,))
        
        questions = cursor.fetchall()
        
        # Get quality metrics if available
        cursor.execute('''
            SELECT AVG(relevance_score), AVG(technical_accuracy), 
                   AVG(communication_clarity), AVG(depth_of_knowledge), AVG(overall_rating)
            FROM quality_metrics 
            WHERE session_id = ?
        ''', (session_id,))
        
        quality_avg = cursor.fetchone()
        
        conn.close()
        
        return {
            "session_info": {
                "session_id": session_row[0],
                "candidate_name": session_row[1],
                "programming_language": session_row[2],
                "experience_level": session_row[3],
                "duration_minutes": session_row[9],
                "questions_asked": session_row[6],
                "final_difficulty": session_row[7],
                "topics_covered": json.loads(session_row[8]) if session_row[8] else []
            },
            "questions_and_responses": [
                {
                    "question": q[0],
                    "category": q[1], 
                    "difficulty": q[2],
                    "candidate_response": q[3],
                    "ai_follow_up": q[4],
                    "timestamp": q[5]
                } for q in questions
            ],
            "quality_metrics": {
                "avg_relevance": round(quality_avg[0], 2) if quality_avg[0] else None,
                "avg_technical_accuracy": round(quality_avg[1], 2) if quality_avg[1] else None,
                "avg_communication": round(quality_avg[2], 2) if quality_avg[2] else None,
                "avg_knowledge_depth": round(quality_avg[3], 2) if quality_avg[3] else None,
                "avg_overall_rating": round(quality_avg[4], 2) if quality_avg[4] else None
            }
        }
    
    def get_performance_trends(self, days: int = 30) -> Dict:
        """Analyze performance trends over time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions per day
        cursor.execute('''
            SELECT DATE(start_time) as date, COUNT(*) as sessions,
                   AVG(questions_asked) as avg_questions,
                   AVG(final_difficulty) as avg_difficulty,
                   AVG(total_duration_minutes) as avg_duration
            FROM interview_sessions 
            WHERE start_time >= datetime('now', '-{} days')
            GROUP BY DATE(start_time)
            ORDER BY date
        '''.format(days))
        
        daily_stats = cursor.fetchall()
        
        # Most common issues/errors
        cursor.execute('''
            SELECT error_type, COUNT(*) as count
            FROM error_logs 
            WHERE error_timestamp >= datetime('now', '-{} days')
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 10
        '''.format(days))
        
        common_errors = cursor.fetchall()
        
        # Language/level distribution
        cursor.execute('''
            SELECT programming_language, experience_level, COUNT(*) as count
            FROM interview_sessions 
            WHERE start_time >= datetime('now', '-{} days')
            GROUP BY programming_language, experience_level
            ORDER BY count DESC
        '''.format(days))
        
        language_distribution = cursor.fetchall()
        
        conn.close()
        
        return {
            "daily_statistics": [
                {
                    "date": row[0],
                    "sessions": row[1],
                    "avg_questions": round(row[2], 1) if row[2] else 0,
                    "avg_difficulty": round(row[3], 1) if row[3] else 0,
                    "avg_duration": round(row[4], 1) if row[4] else 0
                } for row in daily_stats
            ],
            "common_errors": [
                {"error_type": row[0], "count": row[1]} for row in common_errors
            ],
            "language_experience_distribution": [
                {
                    "language": row[0],
                    "experience_level": row[1], 
                    "count": row[2]
                } for row in language_distribution
            ]
        }
    
    def export_for_retraining(self, output_file: str = "interview_data_export.json"):
        """Export interview data for model retraining"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.programming_language, s.experience_level,
                   qr.question_text, qr.question_category, qr.question_difficulty,
                   qr.candidate_response, qr.ai_follow_up
            FROM interview_sessions s
            JOIN question_responses qr ON s.session_id = qr.session_id
            WHERE s.status = 'completed'
            ORDER BY s.start_time
        ''')
        
        training_data = []
        for row in cursor.fetchall():
            training_data.append({
                "instruction": f"You are conducting a technical interview for a {row[1]} {row[0]} developer position.",
                "input": f"Question: {row[2]}\nCandidate Response: {row[5]}\nCategory: {row[3]}\nDifficulty: {row[4]}",
                "output": row[6],
                "metadata": {
                    "language": row[0],
                    "experience_level": row[1],
                    "category": row[3],
                    "difficulty": row[4]
                }
            })
        
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"📊 Exported {len(training_data)} training examples to {output_file}")
        return len(training_data)

# Usage example
if __name__ == "__main__":
    # Example usage
    logger = InterviewLogger()
    
    # Start a test session
    session_id = logger.start_session(
        candidate_name="Test User",
        programming_language="python", 
        experience_level="mid_level"
    )
    
    # Log some Q&A
    logger.log_question_response(
        question_text="What's the difference between lists and tuples?",
        question_category="fundamentals",
        question_difficulty=3,
        candidate_response="Lists are mutable, tuples are immutable",
        ai_follow_up="Good! Can you give me an example of when you'd use each?",
        response_time_seconds=15.5
    )
    
    # End session
    logger.end_session(final_difficulty=4)
    
    # Analyze data
    analyzer = InterviewAnalyzer()
    summary = analyzer.get_session_summary(session_id)
    print(json.dumps(summary, indent=2)) 