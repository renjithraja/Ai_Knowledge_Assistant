"""
Uses an LLM (via Groq API) to generate answers from retrieved context.
"""

from groq import Groq


class ReasoningAgent:
    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        # Initialize Groq client and model configuration
        self.client = Groq(api_key=api_key)
        self.model = model

    def answer(self, query, context):
        """
        Generate a response by combining the user query with retrieved context.
        """
        prompt = (
            f"Answer the user's question using the following context:\n\n"
            f"{context}\n\nQuestion: {query}\nAnswer:"
        )

        # Query the LLM for reasoning and response generation
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.3,
            max_tokens=512,
        )

        # Return the model's text output
        return response.choices[0].message.content.strip()
