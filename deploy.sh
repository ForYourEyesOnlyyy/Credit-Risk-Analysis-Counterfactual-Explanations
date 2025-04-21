#!/bin/bash

# Check if the required model directories exist
if [ ! -d "models/encoder/" ] || [ ! -d "models/loan_grader/" ] || [ ! -d "models/model/" ] || [ ! -d "models/scaler/" ]; then
  echo "Error: One or more required model directories are missing."
  exit 1
fi

# Check if the processed data directory exists and is not empty
if [ ! -d "data/processed" ] || [ -z "$(ls -A data/processed)" ]; then
  echo "Error: The data/processed directory is missing or empty."
  exit 1
fi

# Run the Streamlit app
PYTHONPATH=. streamlit run deploy/app.py