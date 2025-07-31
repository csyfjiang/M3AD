#!/bin/bash

# Set session name and conda environment
SESSION_NAME="swin_training_nine_label"
CONDA_ENV="CONDA_ENVS"  # Your conda environment name

# Check if tmux session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching to it..."
    tmux attach-session -t $SESSION_NAME
else
    echo "Creating new tmux session: $SESSION_NAME"
    
    # Create new tmux session (run in background)
    tmux new-session -d -s $SESSION_NAME
    
    # Activate conda environment
    tmux send-keys -t $SESSION_NAME "conda activate $CONDA_ENV" Enter
    
    # Wait to ensure environment is activated
    sleep 2
    
    # Send Python training command to session
    tmux send-keys -t $SESSION_NAME "python train_nine_label.py" Enter
    
    echo "Training started in tmux session: $SESSION_NAME"
    echo "Conda environment '$CONDA_ENV' activated"
    echo "To attach to the session, run: tmux attach-session -t $SESSION_NAME"
    echo "To detach from session: Ctrl+B, then press D" 
    echo "To list all sessions: tmux list-sessions"
fi