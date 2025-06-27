# 📊 ANALYTICS DASHBOARD - View stored interview data and insights

# Install additional packages for analytics
import subprocess
import sys

# Install packages needed for analytics
packages = ['pandas', 'plotly']
for package in packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Add project to path
import sys
sys.path.append('/content/interview-ai')

# Launch analytics dashboard
from analytics_dashboard import create_dashboard

print("📊 Launching Interview Analytics Dashboard...")
print("💡 This dashboard shows:")
print("  - Interview statistics and trends")
print("  - Detailed session analysis")
print("  - Data export for model retraining")

# Create and launch dashboard on different port
dashboard = create_dashboard()
dashboard.launch(share=True, server_port=7862) 