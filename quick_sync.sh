#!/bin/bash
# Quick sync script for Cursor → GitHub

echo "🔄 Syncing changes to GitHub..."

# Add all changes
git add .

# Check if there are changes
if git diff --cached --quiet; then
    echo "💤 No changes to sync"
else
    # Commit with timestamp
    git commit -m "Auto-sync: $(date)"
    
    # Push to GitHub
    git push origin main
    
    echo "✅ Synced to GitHub!"
    echo "🔗 Run 'git pull' in Colab to get updates"
fi
