"""
Command-line entry point for running the multimodal AI knowledge assistant.
Initializes all agents, processes a user query, and returns an answer.
"""

import argparse
import os
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.controller_agent import Controller
from dotenv import load_dotenv

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

if __name__ == "__main__":
    # Parse command-line query input
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Enter your question")
    args = parser.parse_args()

    # Retrieve API key from environment
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Initialize all agents
    intent_agent = IntentAgent(model_path="models/intent_model.pkl")
    retrieval_agent = RetrievalAgent()
    vision_agent = VisionAgent()
    reasoning_agent = ReasoningAgent(api_key=GROQ_API_KEY)

    # Create controller to orchestrate agents
    controller = Controller(intent_agent, retrieval_agent, vision_agent, reasoning_agent)

    # Process query and display final answer
    answer = controller.handle_query(args.query)
    print("\n[Answer]\n", answer)
