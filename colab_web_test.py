# COLAB WEB INTERFACE - Add this cell to your training notebook

# Install Gradio if not already installed
import subprocess
import sys

try:
    import gradio as gr
except ImportError:
    print("Installing Gradio...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
    import gradio as gr

# Add project to Python path
sys.path.append('/content/interview-ai')

# Import the web interface
from web_interface import launch_web_interface

# Launch with public sharing enabled for Colab
print("Launching Technical Interview AI Web Interface...")
print("This will create a public link you can access from anywhere!")

# Launch the interface
launch_web_interface(share=True, port=7860) 