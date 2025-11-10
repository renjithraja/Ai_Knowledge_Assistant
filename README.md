🧠 AI Knowledge Assistant

A multi-agent, multimodal AI system that extracts and retrieves knowledge from PDFs and images, combines LLM reasoning with VLM (vision-language) understanding, and uses an agent-based architecture to deliver intent-driven Retrieval-Augmented Generation (RAG) responses.



📘 Introduction

This project implements an AI Knowledge Assistant capable of understanding user questions and answering them intelligently using information extracted from PDF documents and embedded images.

It classifies user queries into intents (fact, analysis, summary, or visual) and uses multiple specialized agents to retrieve relevant information, analyze charts or figures, and generate structured, context-aware answers.



⚙️ System Architecture

The system follows a multi-agent modular design, where each agent performs a specific task:

Agent	Responsibility

Intent Agent	Classifies user queries into four intents – fact, summary, analysis, and visual.

Retrieval Agent	Performs RAG by retrieving the most relevant text and visual data from the Chroma vector database.

Vision Agent	Uses the BLIP model to describe or interpret images extracted from PDFs.

Reasoning Agent	Uses an LLM (via the Groq API) to generate final, context-aware responses.

Controller Agent	Orchestrates the workflow by routing the query to appropriate agents based on the intent.


🧩 Features

✅ Intent Classification – Automatically detects the type of question (fact, analysis, summary, or visual).

✅ Multimodal RAG – Retrieves both text and image-based information from PDFs.

✅ Visual Understanding – Interprets and describes charts, graphs, and diagrams using CLIP.

✅ LLM Reasoning – Generates accurate, contextual answers using Groq’s Llama model.

✅ Streamlit Chat UI – Interactive chat-like interface for user queries.

✅ Extensible Design – Easily add more PDFs, images, or custom models.



🧠 Tech Stack
Category	Tools/Frameworks

Programming Language	Python 3.10+

Web Framework	Streamlit

LLM API	Groq (Llama 3.3 70B)

VLM	CLIP (For fast, lightweight image understanding and captioning)

Embeddings	SentenceTransformers (all-MiniLM-L6-v2)

Vector Store	ChromaDB

ML Model	Logistic Regression + TF-IDF (for Intent Classification)

PDF Parsing	PyMuPDF

Environment Management	python-dotenv


📂 Project Structure
multi_agent/
│
├── agents/
│   ├── intent_agent.py
│   ├── retrieval_agent.py
│   ├── reasoning_agent.py
│   ├── vision_agent.py
│   └── controller_agent.py
│
├── utils/
│   ├── pdf_utils.py
│   ├── chroma_utils.py
│   └── embed_utils.py
│
├── data/
│   ├── reports/                 # Input PDFs
│   ├── processed/
│   │   ├── text/               # Extracted text
│   │   └── images/             # Extracted images
│   └── intent_data.csv         # Labeled data for training
│
├── models/
│   ├── intent_model.pkl
│   └── chroma_db/
│
├── app.py                      # Streamlit app (chat interface)
├── main.py                     # CLI orchestrator
├── init_chroma.py              # Builds knowledge base
├── train_intent_model.py       # Trains intent classifier                     
├── requirements.txt
└── README.md



🔐 .env File Template

Create a .env file in the project root and add your Groq API key:

GROQ_API_KEY=your_groq_key




🧾 Installation & Setup

1. Clone the Repository
git clone https://github.com/renjithraja/Ai_Knowledge_Assistant.git
cd multi_agent_knowledge_assistant

2. Create Virtual Environment
python -m venv myenv

3. Activate Virtual Environment

Windows:

myenv\Scripts\activate


macOS/Linux:

source myenv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

5. Add Your .env File

Include your Groq API key as shown above.

6. Build Knowledge Base

Make sure your PDFs are in data/reports/, then run:

python init_chroma.py

7. Train Intent Classifier (if not pre-trained)
python train_intent_model.py

8. Test CLI Query
python main.py --query "Explain the chart about water consumption"

9. Launch Streamlit UI
streamlit run app.py


💬 Example Queries

Type	Example

Fact	“What are the key statistics about drought damage in Europe from the Water Overview 2024 report?”

Analysis	“Compare water availability trends in 2023 and 2024.”

Summary	“Summarize the key sustainability goals from the 2023 report.”

Visual	“Explain the chart showing renewable water usage.”


🧾 Notes

To add new PDFs, place them in data/reports/ and rerun init_chroma.py.

Image extraction from PDFs is automatic via PyMuPDF.

If any image cannot be read, the VisionAgent safely skips it.

Works seamlessly in CPU mode; GPU will accelerate BLIP-2 captioning.

👨‍💻 Author

Renjith R
