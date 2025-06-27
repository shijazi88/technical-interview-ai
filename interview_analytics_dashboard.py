#!/usr/bin/env python3
"""
Interview Analytics Dashboard
Web interface for analyzing stored interview data and generating improvement insights
"""

import gradio as gr
import sqlite3
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from interview_logger import InterviewAnalyzer
import os

class AnalyticsDashboard:
    """Web dashboard for interview analytics"""
    
    def __init__(self, db_path: str = "./interview_logs.db"):
        self.db_path = db_path
        self.analyzer = InterviewAnalyzer(db_path)
        
    def get_overview_stats(self):
        """Get high-level overview statistics"""
        if not os.path.exists(self.db_path):
            return "No data available. Run some interviews first!"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM interview_sessions WHERE status = "completed"')
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM question_responses')
        total_questions = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(total_duration_minutes) FROM interview_sessions WHERE status = "completed"')
        avg_duration = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(final_difficulty) FROM interview_sessions WHERE status = "completed"')
        avg_difficulty = cursor.fetchone()[0]
        
        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM interview_sessions 
            WHERE start_time >= datetime('now', '-7 days') AND status = "completed"
        ''')
        recent_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        stats = f"""
## 📊 Interview Analytics Overview

### 🎯 **Overall Statistics**
- **Total Completed Interviews:** {total_sessions}
- **Total Questions Asked:** {total_questions}
- **Average Interview Duration:** {avg_duration:.1f if avg_duration else 0} minutes
- **Average Final Difficulty:** {avg_difficulty:.1f if avg_difficulty else 0}/8

### 📈 **Recent Activity** (Last 7 days)
- **Recent Interviews:** {recent_sessions}
- **Questions per Interview:** {total_questions/total_sessions:.1f if total_sessions > 0 else 0}

### 🎮 **Model Performance**
- **Model Version:** CodeLlama-7b-Instruct-hf
- **Data Collection Status:** {"✅ Active" if total_sessions > 0 else "❌ No data yet"}
"""
        return stats
    
    def get_language_distribution(self):
        """Get programming language distribution chart"""
        if not os.path.exists(self.db_path):
            return None
        
        conn = sqlite3.connect(self.db_path)
        
        # Get language and experience level distribution
        df = pd.read_sql_query('''
            SELECT programming_language, experience_level, COUNT(*) as count
            FROM interview_sessions 
            WHERE status = "completed"
            GROUP BY programming_language, experience_level
            ORDER BY count DESC
        ''', conn)
        
        conn.close()
        
        if df.empty:
            return None
        
        # Create visualization
        fig = px.sunburst(
            df, 
            path=['programming_language', 'experience_level'], 
            values='count',
            title="Interview Distribution by Language and Experience Level"
        )
        
        return fig
    
    def get_performance_trends(self, days: int = 30):
        """Get performance trends over time"""
        if not os.path.exists(self.db_path):
            return None, "No data available"
        
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query(f'''
            SELECT DATE(start_time) as date, 
                   COUNT(*) as sessions,
                   AVG(questions_asked) as avg_questions,
                   AVG(final_difficulty) as avg_difficulty,
                   AVG(total_duration_minutes) as avg_duration
            FROM interview_sessions 
            WHERE start_time >= datetime('now', '-{days} days') 
            AND status = "completed"
            GROUP BY DATE(start_time)
            ORDER BY date
        ''', conn)
        
        conn.close()
        
        if df.empty:
            return None, "No data in the selected time range"
        
        # Create multi-line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['sessions'],
            mode='lines+markers',
            name='Sessions per Day',
            yaxis='y1'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['avg_difficulty'],
            mode='lines+markers',
            name='Average Difficulty',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"Performance Trends (Last {days} days)",
            xaxis_title="Date",
            yaxis=dict(title="Sessions per Day", side="left"),
            yaxis2=dict(title="Average Difficulty", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        summary = f"📈 **Trend Analysis:**\n"
        summary += f"- **Total Days:** {len(df)}\n"
        summary += f"- **Peak Sessions:** {df['sessions'].max():.0f} on {df.loc[df['sessions'].idxmax(), 'date']}\n"
        summary += f"- **Avg Difficulty Range:** {df['avg_difficulty'].min():.1f} - {df['avg_difficulty'].max():.1f}\n"
        
        return fig, summary
    
    def get_detailed_session_analysis(self, session_id: str):
        """Get detailed analysis of a specific session"""
        if not session_id.strip():
            return "Please enter a session ID"
        
        summary = self.analyzer.get_session_summary(session_id)
        
        if "error" in summary:
            return f"❌ {summary['error']}"
        
        # Format the detailed analysis
        session_info = summary['session_info']
        questions = summary['questions_and_responses']
        
        analysis = f"""
## 🔍 **Session Analysis: {session_id[:8]}...**

### 👤 **Interview Details**
- **Candidate:** {session_info['candidate_name']}
- **Language:** {session_info['programming_language']}
- **Experience Level:** {session_info['experience_level']}
- **Duration:** {session_info['duration_minutes']:.1f} minutes
- **Questions Asked:** {session_info['questions_asked']}
- **Final Difficulty:** {session_info['final_difficulty']}/8
- **Topics Covered:** {', '.join(session_info['topics_covered'])}

### 💬 **Question-Response Analysis**
"""
        
        for i, q in enumerate(questions, 1):
            analysis += f"""
**Question {i}:** {q['question']}
- **Category:** {q['category']} | **Difficulty:** {q['difficulty']}/10
- **Candidate Response:** _{q['candidate_response'][:200]}{'...' if len(q['candidate_response']) > 200 else ''}_
- **AI Follow-up:** _{q['ai_follow_up'][:200]}{'...' if len(q['ai_follow_up']) > 200 else ''}_

---
"""
        
        return analysis
    
    def get_common_issues(self):
        """Identify common issues and areas for improvement"""
        if not os.path.exists(self.db_path):
            return "No data available for analysis"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get question categories with low performance
        cursor.execute('''
            SELECT qr.question_category, 
                   COUNT(*) as question_count,
                   AVG(s.final_difficulty) as avg_difficulty,
                   AVG(s.total_duration_minutes) as avg_duration
            FROM question_responses qr
            JOIN interview_sessions s ON qr.session_id = s.session_id
            WHERE s.status = "completed"
            GROUP BY qr.question_category
            ORDER BY avg_difficulty ASC
        ''')
        
        category_performance = cursor.fetchall()
        
        # Get most common error types
        cursor.execute('''
            SELECT error_type, COUNT(*) as count, 
                   MAX(error_timestamp) as last_occurrence
            FROM error_logs 
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 5
        ''')
        
        common_errors = cursor.fetchall()
        
        conn.close()
        
        analysis = "## 🔧 **Areas for Improvement**\n\n"
        
        if category_performance:
            analysis += "### 📊 **Question Category Performance**\n"
            for cat, count, difficulty, duration in category_performance:
                analysis += f"- **{cat.title()}:** {count} questions, avg difficulty {difficulty:.1f}/8, avg duration {duration:.1f}min\n"
        
        analysis += "\n### ❌ **Common Technical Issues**\n"
        if common_errors:
            for error_type, count, last_seen in common_errors:
                analysis += f"- **{error_type}:** {count} occurrences (last: {last_seen[:10]})\n"
        else:
            analysis += "- ✅ No significant errors detected\n"
        
        analysis += "\n### 💡 **Recommendations**\n"
        analysis += "- Focus training on lower-performing question categories\n"
        analysis += "- Add more diverse examples for challenging topics\n"
        analysis += "- Monitor error patterns for system improvements\n"
        analysis += "- Consider adding quality feedback collection\n"
        
        return analysis
    
    def export_training_data(self):
        """Export data for model retraining"""
        try:
            count = self.analyzer.export_for_retraining()
            return f"✅ Successfully exported {count} training examples to 'interview_data_export.json'"
        except Exception as e:
            return f"❌ Export failed: {str(e)}"

def create_analytics_dashboard():
    """Create the Gradio analytics dashboard"""
    dashboard = AnalyticsDashboard()
    
    css = """
    .analytics-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Interview Analytics Dashboard") as interface:
        
        # Header
        gr.HTML("""
        <div class="analytics-header">
            <h1>📊 Interview Analytics Dashboard</h1>
            <p>Analyze your CodeLlama interview model performance and identify areas for improvement</p>
        </div>
        """)
        
        with gr.Tab("📈 Overview"):
            overview_stats = gr.Markdown(dashboard.get_overview_stats())
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Data", variant="primary")
                refresh_btn.click(
                    dashboard.get_overview_stats,
                    outputs=overview_stats
                )
        
        with gr.Tab("📊 Distribution Analysis"):
            lang_chart = gr.Plot(dashboard.get_language_distribution())
            
            with gr.Row():
                days_input = gr.Slider(
                    minimum=7, 
                    maximum=90, 
                    value=30,
                    step=1,
                    label="Analysis Period (Days)"
                )
                
            trends_chart = gr.Plot()
            trends_summary = gr.Markdown()
            
            days_input.change(
                dashboard.get_performance_trends,
                inputs=days_input,
                outputs=[trends_chart, trends_summary]
            )
            
            # Initial load
            interface.load(
                dashboard.get_performance_trends,
                inputs=gr.Number(value=30, visible=False),
                outputs=[trends_chart, trends_summary]
            )
        
        with gr.Tab("🔍 Session Analysis"):
            with gr.Row():
                session_id_input = gr.Textbox(
                    label="Session ID",
                    placeholder="Enter session ID to analyze...",
                    lines=1
                )
                analyze_btn = gr.Button("🔍 Analyze Session", variant="secondary")
            
            session_analysis = gr.Markdown()
            
            analyze_btn.click(
                dashboard.get_detailed_session_analysis,
                inputs=session_id_input,
                outputs=session_analysis
            )
        
        with gr.Tab("🔧 Improvement Insights"):
            issues_analysis = gr.Markdown(dashboard.get_common_issues())
            
            with gr.Row():
                refresh_issues_btn = gr.Button("🔄 Refresh Analysis", variant="primary")
                export_data_btn = gr.Button("📤 Export Training Data", variant="secondary")
            
            export_status = gr.Textbox(label="Export Status", lines=2)
            
            refresh_issues_btn.click(
                dashboard.get_common_issues,
                outputs=issues_analysis
            )
            
            export_data_btn.click(
                dashboard.export_training_data,
                outputs=export_status
            )
        
        # Instructions
        gr.HTML("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f8fafc; border-radius: 8px; border-left: 4px solid #4f46e5;">
            <h3>📋 How to Use This Dashboard:</h3>
            <ul>
                <li><strong>Overview:</strong> See high-level statistics about your interview model</li>
                <li><strong>Distribution:</strong> Analyze language/experience patterns and trends</li>
                <li><strong>Session Analysis:</strong> Deep dive into specific interview sessions</li>
                <li><strong>Improvement Insights:</strong> Identify issues and export data for retraining</li>
            </ul>
            <p><strong>💡 Pro Tip:</strong> Use the exported training data to retrain your model with real-world examples!</p>
        </div>
        """)
    
    return interface

def launch_analytics_dashboard(share=False, port=7861):
    """Launch the analytics dashboard"""
    print("📊 Starting Interview Analytics Dashboard...")
    
    interface = create_analytics_dashboard()
    
    print(f"🌐 Dashboard will be available at:")
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
    
    parser = argparse.ArgumentParser(description="Interview Analytics Dashboard")
    parser.add_argument("--db_path", type=str, default="./interview_logs.db", 
                       help="Path to interview logs database")
    parser.add_argument("--share", action="store_true", 
                       help="Create public shareable link")
    parser.add_argument("--port", type=int, default=7861, 
                       help="Port to run the dashboard on")
    
    args = parser.parse_args()
    
    launch_analytics_dashboard(share=args.share, port=args.port) 