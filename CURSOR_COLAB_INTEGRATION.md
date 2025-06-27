# 🔗 Cursor + Google Colab Integration Guide

This guide shows you **4 different ways** to connect Cursor with Google Colab for an optimal development experience.

## 🎯 Why Integrate Cursor + Colab?

- **Write in Cursor:** Superior code editing, AI assistance, debugging
- **Execute on Colab:** Free GPU/TPU access, no local setup required
- **Best of Both:** Professional IDE + Cloud computing power

---

## 🚀 Method 1: Local Runtime Connection (Recommended)

**Write in Cursor, execute on Colab's GPUs with real-time sync.**

### Setup Steps:

1. **In Cursor Terminal:**

```bash
# Install Jupyter with WebSocket support
pip install jupyter jupyter_http_over_ws

# Enable the extension
jupyter serverextension enable --py jupyter_http_over_ws

# Start local Jupyter server
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0 \
  --no-browser
```

2. **In Google Colab:**

   - Click the dropdown arrow next to "Connect"
   - Select "Connect to local runtime"
   - Enter: `http://localhost:8888/?token=your_token`

3. **Workflow:**
   - Edit code in Cursor
   - Save files
   - Execute cells in Colab (connected to your local environment)
   - Results appear in Colab but run on Colab's GPUs

### ✅ Pros:

- Real-time code sync
- Use Colab's GPU/TPU
- Keep Cursor's superior editing experience
- No file upload/download needed

### ❌ Cons:

- Requires stable internet connection
- Local Jupyter server must stay running

---

## 📁 Method 2: File Synchronization

**Develop in Cursor, sync files to Colab manually or automatically.**

### Quick Setup:

```bash
# Run the integration script
python colab_cursor_integration.py
```

### Manual Sync Process:

1. **Package your project:**

```bash
# In Cursor terminal
tar -czf project_files.tar.gz *.py *.json *.md

# Upload to Google Drive manually
```

2. **In Google Colab:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract project files
!tar -xzf /content/drive/MyDrive/project_files.tar.gz

# Run your training
!python colab_training_pipeline.py --num_scenarios 100
```

### Automated Sync:

```bash
# Use the provided sync script
./sync_colab.sh upload    # Upload to Colab
./sync_colab.sh download  # Download results
```

### ✅ Pros:

- Simple setup
- Works with any file sharing method
- Good for one-way transfers

### ❌ Cons:

- Manual sync required
- Not real-time
- File management overhead

---

## 🌐 Method 3: GitHub Integration

**Use GitHub as a bridge between Cursor and Colab.**

### Setup:

1. **In Cursor:**

```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

2. **In Google Colab:**

```python
# Clone your repository
!git clone https://github.com/yourusername/your-repo.git
%cd your-repo

# Run your code
!python colab_training_pipeline.py
```

3. **Update Workflow:**

```bash
# In Cursor: Make changes and push
git add .
git commit -m "Update training script"
git push

# In Colab: Pull latest changes
!git pull origin main
```

### ✅ Pros:

- Version control included
- Easy collaboration
- Public/private repository options
- Direct Colab integration

### ❌ Cons:

- Requires Git knowledge
- Extra steps for each update
- Internet dependent

---

## 🔌 Method 4: VS Code/Cursor Extensions

**Use Cursor's VS Code compatibility with Colab extensions.**

### Required Extensions:

1. **Install in Cursor:**

   - `Python` extension
   - `Jupyter` extension
   - `Remote - SSH` extension (if using SSH)

2. **Configure Cursor:**

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "/usr/bin/python3",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "jupyter.alwaysTrustNotebooks": true,
  "files.associations": {
    "*.ipynb": "jupyter-notebook"
  },
  "jupyter.sendSelectionToInteractiveWindow": true
}
```

3. **Create Tasks:**

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Upload to Colab",
      "type": "shell",
      "command": "./sync_colab.sh upload"
    },
    {
      "label": "Start Jupyter Server",
      "type": "shell",
      "command": "python colab_workflow.py jupyter"
    }
  ]
}
```

### ✅ Pros:

- Native notebook support in Cursor
- Keyboard shortcuts and tasks
- Integrated terminal

### ❌ Cons:

- Extension compatibility varies
- Setup complexity

---

## ⚡ Quick Start Scripts

### Automatic Setup:

```bash
# Run the complete integration setup
python colab_cursor_integration.py
```

### Common Workflows:

```bash
# Start Jupyter server for local runtime
python colab_workflow.py jupyter

# Sync files to Google Drive
python colab_workflow.py sync

# Convert Python script to notebook
python colab_workflow.py convert your_script.py

# Package project for upload
./sync_colab.sh upload
```

---

## 🛠️ Development Workflows

### Workflow 1: Real-time Development

1. Start local Jupyter server in Cursor
2. Connect Colab to local runtime
3. Edit code in Cursor
4. Execute in Colab with GPU acceleration
5. See results instantly

### Workflow 2: Batch Processing

1. Develop complete scripts in Cursor
2. Test locally with small datasets
3. Upload to Colab for full training
4. Monitor progress in Colab
5. Download results

### Workflow 3: Collaborative Development

1. Use GitHub integration
2. Multiple developers work in Cursor
3. Push changes to repository
4. Team members run on Colab
5. Share results via notebooks

---

## 🎯 Best Practices

### For Local Runtime:

- Keep Jupyter server running in background
- Use stable internet connection
- Monitor memory usage in both environments

### For File Sync:

- Use `.gitignore` for large files
- Compress files before upload
- Automate sync with scripts

### For GitHub Integration:

- Use meaningful commit messages
- Keep repository clean
- Use branches for experiments

### For Extensions:

- Keep Cursor updated
- Configure Python interpreter correctly
- Use keyboard shortcuts for efficiency

---

## 🔧 Troubleshooting

### Common Issues:

**1. Local Runtime Connection Failed**

```bash
# Check if Jupyter is running
ps aux | grep jupyter

# Restart with correct settings
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888
```

**2. File Sync Issues**

```bash
# Check file permissions
ls -la *.py

# Recreate archive
tar -czf project_files.tar.gz *.py --verbose
```

**3. Extension Problems**

- Restart Cursor
- Reinstall Python extension
- Check Python interpreter path

**4. Memory Issues**

- Close unnecessary tabs
- Clear Jupyter kernel
- Restart Colab runtime

---

## 📊 Comparison Table

| Method        | Setup Time | Real-time  | GPU Access   | Ease of Use | Best For           |
| ------------- | ---------- | ---------- | ------------ | ----------- | ------------------ |
| Local Runtime | 5 min      | ✅ Yes     | ✅ Colab GPU | ⭐⭐⭐⭐    | Active development |
| File Sync     | 2 min      | ❌ No      | ✅ Colab GPU | ⭐⭐⭐      | Batch processing   |
| GitHub        | 10 min     | ❌ No      | ✅ Colab GPU | ⭐⭐⭐⭐⭐  | Team collaboration |
| Extensions    | 15 min     | ⚡ Partial | ✅ Colab GPU | ⭐⭐        | Power users        |

---

## 🚀 Getting Started Now

### Option 1: Quick Start (Local Runtime)

```bash
# In Cursor terminal
pip install jupyter jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
python colab_workflow.py jupyter
```

Then connect Colab to `http://localhost:8888`

### Option 2: File Sync Setup

```bash
# In Cursor terminal
python colab_cursor_integration.py
./sync_colab.sh upload
```

Then extract files in Colab and run your training.

### Option 3: GitHub Integration

```bash
# In Cursor terminal
git init && git add . && git commit -m "Initial"
git remote add origin https://github.com/username/repo.git
git push -u origin main
```

Then clone in Colab and run.

---

## 💡 Pro Tips

1. **Use Local Runtime for Development:** Real-time feedback while coding
2. **Use File Sync for Training:** Long-running processes work better
3. **Use GitHub for Collaboration:** Version control + easy sharing
4. **Combine Methods:** Different approaches for different phases

5. **Keyboard Shortcuts in Cursor:**

   - `Ctrl+Shift+P` → Search for "Upload to Colab" task
   - `Ctrl+` ` → Open integrated terminal
   - `F5` → Run current Python file

6. **Colab Optimization:**
   - Use `!pip install` once at the beginning
   - Save checkpoints frequently
   - Monitor GPU usage with `!nvidia-smi`

---

## 🎉 Success! You're Ready

You now have **4 different ways** to connect Cursor with Google Colab. Choose the method that best fits your workflow:

- **Quick prototyping:** Local Runtime
- **Large scale training:** File Sync
- **Team development:** GitHub Integration
- **Advanced users:** Extensions

**Start with Local Runtime for the best experience!** 🚀

```bash
python colab_workflow.py jupyter
```

Then connect your Colab notebook to `http://localhost:8888` and enjoy the best of both worlds! 🌟
