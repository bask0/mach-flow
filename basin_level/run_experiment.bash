#!/bin/bash

# Initialize the model variable
model=""

# Function to display usage
usage() {
    echo "Usage: $0 -m [LSTM|TCN]"
    exit 1
}

# Parse arguments
while getopts ":m:" opt; do
    case $opt in
        m) model=$OPTARG ;;
        *) usage ;;
    esac
done

# Check if model argument is provided
if [ -z "$model" ]; then
    usage
fi

# Validate the model argument
if [ "$model" != "LSTM" ] && [ "$model" != "TCN" ]; then
    echo "Error: Invalid model '$model'. Choose either 'LSTM' or 'TCN'."
    exit 1
fi

# Rest of your script using the model variable
echo "Selected model: $model"

# Define the script path
SCRIPT_PATH="my/script.py"
COMMAND="python ${SCRIPT_PATH} -m ${model}"

for c in {""," -c static_all.yaml"," -c static_dred.yaml"}\
`       `{""," -c expectiles.yaml"}\
`       `{""," -c sqrt_trans.yaml"} ; do
    CMD="${COMMAND}${c}"
    echo "Running: ${CMD}"
    # eval $CMD
done
