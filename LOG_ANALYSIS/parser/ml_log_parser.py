# parser/ml_log_parser.py

import os
import re
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

# Setup
DATA_DIR = "LogHub"
MODEL_FILE = "timestamp_svm_model.pkl"
INPUT_LOG_FILE = "combined_logs.log"
OUTPUT_JSON = "../parsed_logs.json"  # Output in project root

# Regex extractors
def extract_with_regex(line):
    timestamp_pattern = r'(\d{4}[-/]\d{2}[-/]\d{2}[ T]?\d{0,2}:?\d{0,2}:?\d{0,2})'
    log_level_pattern = r'\b(INFO|ERROR|DEBUG|WARN|WARNING|TRACE|FATAL)\b'
    timestamp_match = re.search(timestamp_pattern, line)
    log_level_match = re.search(log_level_pattern, line, flags=re.IGNORECASE)
    timestamp = timestamp_match.group(1) if timestamp_match else None
    log_level = log_level_match.group(1).upper() if log_level_match else None
    return timestamp, log_level

# Load or train model
if os.path.exists(MODEL_FILE):
    model, vectorizer = joblib.load(MODEL_FILE)
    print("Loaded existing model.")
else:
    print("Training model...")
    positive = ['2023-05-14', '12:34:56', '2023/08/11', '11:59:59', '2000-01-01 00:00:00']
    negative = ['ERROR', 'INFO', 'login', 'user', 'failed', 'connected']
    labels = [1]*len(positive) + [0]*len(negative)
    samples = positive + negative
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    X = vectorizer.fit_transform(samples)
    X, labels = shuffle(X, labels, random_state=42)
    model = SVC(probability=True)
    model.fit(X, labels)
    joblib.dump((model, vectorizer), MODEL_FILE)
    print("Model trained and saved.")

# Read input log file
with open(INPUT_LOG_FILE, "r", errors='ignore') as f:
    lines = f.readlines()

# Extract
extracted = []

for line in tqdm(lines, desc="Parsing log lines"):
    timestamp, log_level = extract_with_regex(line)
    if timestamp and log_level:
        extracted.append({'line': line.strip(), 'timestamp': timestamp, 'log_level': log_level})
    else:
        tokens = line.strip().split()
        for token in tokens:
            x = vectorizer.transform([token])
            prob = model.predict_proba(x)[0, 1]
            if prob > 0.5:
                extracted.append({'line': line.strip(), 'timestamp': token, 'log_level': log_level or 'UNKNOWN'})
                break

# Save output
df = pd.DataFrame(extracted)
df.to_json(OUTPUT_JSON, orient='records', lines=True)
print(f"Saved {len(df)} parsed log entries to {OUTPUT_JSON}")
