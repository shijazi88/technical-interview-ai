# 📊 SETUP PROGRESS TRACKING - Install widgets for real-time training progress

print("📦 Installing progress tracking widgets...")

# Install ipywidgets for progress bars and real-time updates
import subprocess
import sys

try:
    import ipywidgets
    print("✅ ipywidgets already installed")
except ImportError:
    print("Installing ipywidgets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ipywidgets"])
    print("✅ ipywidgets installed successfully!")

# Enable widgets extension
try:
    from IPython.display import display, HTML
    print("✅ IPython display ready")
    
    # Test widget functionality
    import ipywidgets as widgets
    test_widget = widgets.FloatProgress(value=0, min=0, max=100, description='Test:')
    print("✅ Widget system working!")
    
except Exception as e:
    print(f"⚠️ Widget setup issue: {e}")
    print("Progress will be shown as text output instead")

print("\n🎯 Progress Tracking Ready!")
print("Your training will show:")
print("  📊 Real-time progress bar")
print("  ⏱️ Elapsed time and remaining time estimates")  
print("  📉 Current loss and best loss tracking")
print("  🔥 Steps per second and ETA")
print("  📈 Live metrics dashboard")

print("\n✅ Ready to start training with visual progress tracking!") 