"""
Trains and saves the intent classification model using labeled text data.
"""

import pandas as pd
from agents.intent_agent import IntentAgent
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load training data from CSV
df = pd.read_csv("data/intent_data.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()

# Initialize and train the intent classification agent
agent = IntentAgent()
agent.train(texts, labels, "models/intent_model.pkl")

print("Intent model trained successfully with 120 examples.")
