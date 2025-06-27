#!/usr/bin/env python3
"""
Auto-Watch: Automatically sync changes to GitHub when files change
Near real-time synchronization for Cursor → Colab
"""

import time
import subprocess
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class AutoSyncHandler(FileSystemEventHandler):
    """Handle file change events and auto-sync to GitHub"""
    
    def __init__(self):
        self.last_sync = 0
        self.sync_delay = 10  # seconds between syncs
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        # Watch Python files
        if event.src_path.endswith('.py'):
            current_time = time.time()
            
            # Debounce - only sync every 10 seconds max
            if current_time - self.last_sync > self.sync_delay:
                self.last_sync = current_time
                filename = Path(event.src_path).name
                print(f"📝 {filename} changed - syncing...")
                self.sync_to_github(f"Auto-update: {filename}")
    
    def sync_to_github(self, message):
        """Sync changes to GitHub"""
        try:
            # Add all changes
            subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
            
            # Check if there are changes
            result = subprocess.run(['git', 'diff', '--cached'], capture_output=True, text=True)
            if result.stdout.strip():
                # Commit changes
                subprocess.run(['git', 'commit', '-m', message], check=True, capture_output=True)
                
                # Push to GitHub
                subprocess.run(['git', 'push', 'origin', 'main'], check=True, capture_output=True)
                
                print(f"✅ Synced: {message}")
                print("🔗 Run !git pull in Colab to get updates")
            else:
                print("💤 No changes to sync")
                
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Sync failed: {e}")

def start_watching():
    """Start watching files for changes"""
    print("🎯 AUTO-WATCH: Real-time Cursor → GitHub Sync")
    print("=" * 50)
    print("👀 Watching Python files for changes...")
    print("🔄 Auto-sync to GitHub every 10 seconds")
    print("⏹️ Press Ctrl+C to stop")
    print()
    
    # Check if git is set up
    if not os.path.exists('.git'):
        print("❌ No git repository found!")
        print("Run: git init && git remote add origin https://github.com/shijazi88/technical-interview-ai")
        return
    
    event_handler = AutoSyncHandler()
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 Auto-watch stopped")
    
    observer.join()

if __name__ == "__main__":
    start_watching() 