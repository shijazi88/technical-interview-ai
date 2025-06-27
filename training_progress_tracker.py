#!/usr/bin/env python3
"""
Training Progress Tracker
Real-time progress monitoring with time estimates and visual feedback
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from transformers import TrainerCallback

class TrainingProgressTracker:
    """Real-time training progress tracker with time estimates"""
    
    def __init__(self, total_steps: int = 1000, update_interval: float = 2.0):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.last_update_time = None
        self.update_interval = update_interval
        
        # Training metrics
        self.current_loss = 0.0
        self.current_epoch = 0
        self.learning_rate = 0.0
        self.best_loss = float('inf')
        
        # Progress tracking
        self.step_times = []
        self.loss_history = []
        self.running = False
        
        # UI elements
        self.progress_widget = None
        self.status_widget = None
        self.metrics_widget = None
        self.setup_widgets()
    
    def setup_widgets(self):
        """Setup IPython widgets for progress display"""
        try:
            # Progress bar
            self.progress_widget = widgets.FloatProgress(
                value=0,
                min=0,
                max=100,
                description='Training:',
                bar_style='info',
                style={'bar_color': '#4CAF50'},
                orientation='horizontal'
            )
            
            # Status text
            self.status_widget = widgets.HTML(
                value="<b>Initializing training...</b>"
            )
            
            # Metrics display
            self.metrics_widget = widgets.HTML(
                value="<div style='font-family: monospace; background: #f5f5f5; padding: 10px; border-radius: 5px;'>Waiting for training to start...</div>"
            )
            
        except ImportError:
            # Fallback for environments without ipywidgets
            self.progress_widget = None
            self.status_widget = None
            self.metrics_widget = None
    
    def start_training(self, total_steps: int, total_epochs: int = 3):
        """Start training progress tracking"""
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.current_step = 0
        self.current_epoch = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.running = True
        self.step_times = []
        self.loss_history = []
        
        print("🚀 Training Progress Tracker Started")
        print(f"📊 Total Steps: {total_steps} | Epochs: {total_epochs}")
        print("=" * 60)
        
        # Display widgets if available
        if self.progress_widget:
            display(self.progress_widget)
            display(self.status_widget)
            display(self.metrics_widget)
    
    def update_progress(self, step: int, epoch: int, loss: float, lr: float = None):
        """Update training progress"""
        if not self.running:
            return
        
        self.current_step = step
        self.current_epoch = epoch
        self.current_loss = loss
        if lr:
            self.learning_rate = lr
        
        # Track best loss
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Record timing
        current_time = time.time()
        self.step_times.append(current_time - self.last_update_time)
        self.loss_history.append(loss)
        self.last_update_time = current_time
        
        # Keep only recent history (last 50 steps)
        if len(self.step_times) > 50:
            self.step_times = self.step_times[-50:]
            self.loss_history = self.loss_history[-50:]
        
        # Update display
        self._update_display()
    
    def _update_display(self):
        """Update the progress display"""
        # Calculate progress percentage
        progress_pct = (self.current_step / self.total_steps) * 100
        
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed_time)
        
        # Estimate remaining time
        if self.current_step > 0 and len(self.step_times) > 5:
            avg_step_time = sum(self.step_times[-10:]) / len(self.step_times[-10:])
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = remaining_steps * avg_step_time
            remaining_str = self._format_time(estimated_remaining)
            eta_time = datetime.now() + timedelta(seconds=estimated_remaining)
            eta_str = eta_time.strftime("%H:%M:%S")
        else:
            remaining_str = "Calculating..."
            eta_str = "Calculating..."
        
        # Calculate loss trend
        if len(self.loss_history) >= 10:
            recent_avg = sum(self.loss_history[-5:]) / 5
            older_avg = sum(self.loss_history[-10:-5]) / 5
            loss_trend = "📉" if recent_avg < older_avg else "📈" if recent_avg > older_avg else "➡️"
        else:
            loss_trend = "➡️"
        
        # Update widgets if available
        if self.progress_widget:
            self.progress_widget.value = progress_pct
            self.progress_widget.description = f'Step {self.current_step}/{self.total_steps}'
            
            # Update status
            self.status_widget.value = f"""
            <div style='font-size: 14px; font-weight: bold; color: #2E7D32;'>
                🔥 Training Epoch {self.current_epoch}/{self.total_epochs} | 
                ⏱️ Elapsed: {elapsed_str} | 
                ⏳ Remaining: {remaining_str} | 
                🎯 ETA: {eta_str}
            </div>
            """
            
            # Update metrics
            self.metrics_widget.value = f"""
            <div style='font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                    <div>
                        <strong>📊 Progress Metrics</strong><br/>
                        Progress: {progress_pct:.1f}%<br/>
                        Current Step: {self.current_step:,}<br/>
                        Steps/sec: {1/sum(self.step_times[-5:])*5:.2f if len(self.step_times) >= 5 else 'N/A'}
                    </div>
                    <div>
                        <strong>📈 Loss Metrics</strong><br/>
                        Current Loss: {self.current_loss:.4f}<br/>
                        Best Loss: {self.best_loss:.4f} {loss_trend}<br/>
                        Learning Rate: {self.learning_rate:.2e if self.learning_rate else 'N/A'}
                    </div>
                </div>
            </div>
            """
        
        # Console output (for non-widget environments)
        else:
            print(f"\r🔥 Step {self.current_step}/{self.total_steps} ({progress_pct:.1f}%) | "
                  f"⏱️ {elapsed_str} | ⏳ {remaining_str} | 📉 Loss: {self.current_loss:.4f}", end="", flush=True)
    
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
    
    def complete_training(self):
        """Mark training as complete"""
        self.running = False
        total_time = time.time() - self.start_time
        
        # Final update
        if self.progress_widget:
            self.progress_widget.value = 100
            self.progress_widget.bar_style = 'success'
            self.status_widget.value = f"""
            <div style='font-size: 16px; font-weight: bold; color: #1B5E20;'>
                🎉 Training Completed! | ⏱️ Total Time: {self._format_time(total_time)} | 
                📉 Final Loss: {self.current_loss:.4f} | 🏆 Best Loss: {self.best_loss:.4f}
            </div>
            """
        
        print(f"\n\n🎉 Training Completed Successfully!")
        print(f"⏱️ Total Training Time: {self._format_time(total_time)}")
        print(f"📊 Total Steps: {self.current_step}")
        print(f"📉 Final Loss: {self.current_loss:.4f}")
        print(f"🏆 Best Loss: {self.best_loss:.4f}")
        print("="*60)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        if not self.start_time:
            return {"error": "Training not started"}
        
        elapsed_time = time.time() - self.start_time if self.running else (self.last_update_time - self.start_time)
        
        return {
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "progress_percentage": (self.current_step / self.total_steps) * 100,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": self._format_time(elapsed_time),
            "current_loss": self.current_loss,
            "best_loss": self.best_loss,
            "average_step_time": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            "loss_trend": "improving" if len(self.loss_history) > 10 and self.loss_history[-1] < self.loss_history[-10] else "stable",
            "status": "running" if self.running else "completed"
        }

class ProgressTrainerCallback(TrainerCallback):
    """Hugging Face Trainer callback for progress tracking"""
    
    def __init__(self, progress_tracker: TrainingProgressTracker):
        self.progress_tracker = progress_tracker
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called when training begins"""
        total_steps = state.max_steps
        total_epochs = int(args.num_train_epochs)
        self.progress_tracker.start_training(total_steps, total_epochs)
    
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        """Called at the end of each training step"""
        if logs:
            current_loss = logs.get('loss', 0.0)
            learning_rate = logs.get('learning_rate', 0.0)
            
            self.progress_tracker.update_progress(
                step=state.global_step,
                epoch=int(state.epoch) if state.epoch else 0,
                loss=current_loss,
                lr=learning_rate
            )
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called when training ends"""
        self.progress_tracker.complete_training()

# Standalone progress monitor for web interface
class WebProgressMonitor:
    """Web-based progress monitor using simple HTML updates"""
    
    def __init__(self):
        self.tracker = None
        self.monitoring = False
    
    def start_monitoring(self, progress_tracker: TrainingProgressTracker):
        """Start monitoring a training session"""
        self.tracker = progress_tracker
        self.monitoring = True
        
        # Create initial HTML display
        display(HTML("""
        <div id="training-progress" style="
            border: 2px solid #4CAF50; 
            border-radius: 10px; 
            padding: 20px; 
            margin: 10px 0;
            background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
            font-family: 'Segoe UI', Arial, sans-serif;
        ">
            <h3 style="color: #2E7D32; margin-top: 0;">🚀 CodeLlama Training Progress</h3>
            <div id="progress-bar" style="
                width: 100%; 
                height: 25px; 
                background-color: #E0E0E0; 
                border-radius: 12px; 
                overflow: hidden;
                margin: 15px 0;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div id="progress-fill" style="
                    height: 100%; 
                    background: linear-gradient(90deg, #4CAF50 0%, #66BB6A 100%); 
                    width: 0%; 
                    transition: width 0.5s ease;
                    border-radius: 12px;
                "></div>
            </div>
            <div id="progress-text" style="
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 20px; 
                font-size: 14px;
                color: #424242;
            ">
                <div>⏱️ <strong>Elapsed:</strong> Initializing...</div>
                <div>⏳ <strong>Remaining:</strong> Calculating...</div>
                <div>📊 <strong>Step:</strong> 0/0</div>
                <div>📉 <strong>Loss:</strong> N/A</div>
            </div>
        </div>
        
        <script>
        function updateProgress() {
            // This will be updated by Python
        }
        </script>
        """))
    
    def update_display_html(self, summary: Dict[str, Any]):
        """Update the HTML display with current progress"""
        if not self.monitoring:
            return
        
        progress_pct = summary.get('progress_percentage', 0)
        
        # Update HTML elements using JavaScript
        display(HTML(f"""
        <script>
        document.getElementById('progress-fill').style.width = '{progress_pct:.1f}%';
        document.getElementById('progress-text').innerHTML = `
            <div>⏱️ <strong>Elapsed:</strong> {summary.get('elapsed_time_formatted', 'N/A')}</div>
            <div>⏳ <strong>Remaining:</strong> {summary.get('estimated_remaining', 'Calculating...')}</div>
            <div>📊 <strong>Step:</strong> {summary.get('current_step', 0):,}/{summary.get('total_steps', 0):,}</div>
            <div>📉 <strong>Loss:</strong> {summary.get('current_loss', 0):.4f}</div>
        `;
        </script>
        """))

# Usage example
if __name__ == "__main__":
    # Example usage
    tracker = TrainingProgressTracker(total_steps=1000)
    
    # Simulate training
    import time
    import random
    
    tracker.start_training(1000, 3)
    
    for step in range(1, 1001):
        # Simulate training step
        time.sleep(0.1)  # Simulate processing time
        
        # Simulate decreasing loss
        loss = 2.0 * (1 - step/1000) + random.uniform(-0.1, 0.1)
        epoch = (step // 334) + 1
        
        tracker.update_progress(step, epoch, loss, 2e-5)
        
        if step % 100 == 0:
            summary = tracker.get_training_summary()
            print(f"\nStep {step} Summary:", summary)
    
    tracker.complete_training() 