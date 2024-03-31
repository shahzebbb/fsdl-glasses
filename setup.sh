#!/bin/bash

# Check if '.' is in PYTHONPATH
if [[ ":$PYTHONPATH:" != *":."* ]]; then

    # Add to ~/.bashrc
    echo 'export PYTHONPATH="$PYTHONPATH:."' >> ~/.bashrc
    echo "Added command to ~/.bashrc"
    source ~/.bashrc
else
    echo "'.' is already in PYTHONPATH"
fi