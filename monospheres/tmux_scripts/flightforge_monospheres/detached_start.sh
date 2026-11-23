#!/bin/bash
# Start the session in the background without attaching to it or touching the calling terminal
# (for scripted/automated runs). Use start.sh for normal interactive use.
# Attach later with: tmux -L mrs a -t simulation

SCRIPT=$(readlink -f $0)
SCRIPTPATH=`dirname $SCRIPT`
cd "$SCRIPTPATH"

export TMUX_SESSION_NAME=simulation
export TMUX_SOCKET_NAME=mrs

# unset TMUX so tmuxinator does not try to switch the current client
TMUX= tmuxinator start -p ./session.yml
