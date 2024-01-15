#!/bin/bash

# Initialize the model variable
model=""

# Function to display usage
usage() {
    echo "Usage: $0 -m [LSTM|TCN|MHA]"
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
if [ "$model" != "LSTM" ] && [ "$model" != "TCN" ] && [ "$model" != "MHA" ]; then
    echo "Error: Invalid model '$model'. Choose either 'LSTM', 'TCN', or 'MHA'."
    exit 1
fi

# Rest of your script using the model variable
echo "Selected model: $model"

# Define the script path
SCRIPT_PATH="basin_level/cli.py"
COMMAND="python ${SCRIPT_PATH} --model ${model}"

for c in {""," -c basin_level/staticall.yaml"," -c basin_level/staticdred.yaml"}\
`       `{""," -c basin_level/allbasins.yaml"}\
`       `{""," -c basin_level/sqrttrans.yaml"} ; do
    CMD="${COMMAND}${c}"
    echo "Running: ${CMD}"
    eval $CMD
done
