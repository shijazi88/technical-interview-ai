# Analytics Dashboard for Interview Data

import gradio as gr
import sqlite3
import json
import os
from interview_logger import InterviewAnalyzer

class SimpleDashboard:
    def __init__(self, db_path="./interview_logs.db"):
        self.db_path = db_path
        self.analyzer = InterviewAnalyzer(db_path)
    
    def get_stats(self):
        if not os.path.exists(self.db_path):
            return "No data available. Run some interviews first!"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM interview_sessions WHERE status = "completed"')
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM question_responses')
        total_questions = cursor.fetchone()[0]
        
        conn.close()
        
        return f"""
# Interview Analytics

## Statistics
- Total Interviews: {total_sessions}
- Total Questions: {total_questions}
- Questions per Interview: {total_questions/total_sessions:.1f if total_sessions > 0 else 0}
"""
    
    def analyze_session(self, session_id):
        if not session_id.strip():
            return "Please enter a session ID"
        
        summary = self.analyzer.get_session_summary(session_id)
        if "error" in summary:
            return f"Error: {summary['error']}"
        
        info = summary['session_info']
        return f"""
# Session Analysis

**Candidate:** {info['candidate_name']}
**Language:** {info['programming_language']}
**Level:** {info['experience_level']}
**Questions:** {info['questions_asked']}
**Duration:** {info['duration_minutes']:.1f} minutes
"""

def create_dashboard():
    dashboard = SimpleDashboard()
    
    with gr.Blocks() as interface:
        gr.HTML("<h1>📊 Interview Analytics Dashboard</h1>")
        
        with gr.Tab("Overview"):
            stats = gr.Markdown(dashboard.get_stats())
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(dashboard.get_stats, outputs=stats)
        
        with gr.Tab("Session Analysis"):
            session_input = gr.Textbox(label="Session ID")
            analyze_btn = gr.Button("Analyze")
            analysis_output = gr.Markdown()
            
            analyze_btn.click(
                dashboard.analyze_session,
                inputs=session_input,
                outputs=analysis_output
            )
    
    return interface

if __name__ == "__main__":
    interface = create_dashboard()
    interface.launch(share=True, server_port=7861) 