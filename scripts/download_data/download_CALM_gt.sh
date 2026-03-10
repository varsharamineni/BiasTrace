#!/bin/bash

# ------------------------
# CALM GT Files Download
# ------------------------

# Set target directory
TARGET_DIR=~/home/vramineni/bias-reasoning-LLM/datasets/CALM_gt

# Create directory if it doesn't exist
mkdir -p $TARGET_DIR
cd $TARGET_DIR || exit

# Base URL of GT files in the repo
BASE_URL="https://raw.githubusercontent.com/vipulgupta1011/CALM/c56e5c725dbb1138806b1642497bddc9a878551d/evaluation/gt"

# List of GT files
FILES=(
  "gt_nli_gender.csv"
  "gt_nli_race.csv"
  "gt_qa_gender.csv"
  "gt_qa_race.csv"
  "gt_sentiment_gender.csv"
  "gt_sentiment_race.csv"
)

# Download each file
for FILE in "${FILES[@]}"; do
    echo "Downloading $FILE..."
    wget -q --show-progress "$BASE_URL/$FILE"
done

echo "All GT files downloaded to $TARGET_DIR"
