"""
Coordinates all agents to handle user queries based on intent.
"""

from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent


class Controller:
    def __init__(self, intent_agent: IntentAgent,
                 retrieval_agent: RetrievalAgent,
                 vision_agent: VisionAgent,
                 reasoning_agent: ReasoningAgent):
        # Initialize all agent components
        self.intent = intent_agent
        self.retrieval = retrieval_agent
        self.vision = vision_agent
        self.reason = reasoning_agent

    def handle_query(self, query: str) -> str:
        """
        Classify query intent and route it to the right agent pipeline.
        """
        # Detect intent type (fact, summary, analysis, visual)
        intent = self.intent.predict(query)
        print(f"[Intent] → {intent}")

        # Handle factual, analytical, or summary queries using text retrieval
        if intent in ["fact", "summary", "analysis"]:
            docs = self.retrieval.query(query, n_results=5)
            context = "\n".join(d["text"] for d in docs)
            return self.reason.answer(query, context)

        # Handle visual queries using image captions
        elif intent == "visual":
            docs = self.retrieval.query(query, n_results=3)
            img_contexts = [
                d["metadata"]["source"]
                for d in docs if "image" in d["metadata"]["source"]
            ]
            captions = [self.vision.describe_image(p) for p in img_contexts]
            context = "\n".join(captions)
            return self.reason.answer(query, context)

        # Fallback if no intent is matched
        return "Intent not recognized."
