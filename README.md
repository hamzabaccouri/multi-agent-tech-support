## Agentic Technical Support System
A multi-agent AI-powered technical support system that simulates interaction between a user and two collaborative AI agents. The system uses a first-line support agent (Agent A) for user interaction and a technical expert agent (Agent B) with RAG capabilities for detailed technical solutions.

## Project Overview
This system consists of:
Agent A: First-line support agent that interacts with users and determines when technical expertise is needed
Agent B: A technical expert agent leveraging Retrieval-Augmented Generation (RAG) utilizes a Chroma vector database in combination with LangChain to access relevant information from a knowledge base. The knowledge base consists of synthetic Q&A data generated by a custom script and stored locally in the TechAssistDB directory.
Synthetic Data Generator: Creates bilingual (EN/FR) technical support Q&A datasets
Streamlit Interface: User-friendly web interface with chat history and session management

## Installation
1. Clone the Repository:  git clone https://github.com/hamzabaccouri/multi-agent-tech-support.git
cd multi-agent-tech-support 
2. Create and activate a virtual environment:
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
3. Install Required Dependencies
Install the necessary Python libraries: pip install -r requirements.txt
4. Set up environment variables:
Create a .env file in the root directory with: 
OPENAI_API_KEY=your_api_key_here

## Optional 
Generate the Knowledge Base (Optional)
  .If you want to regenerate the synthetic Q&A data, run the following command:
   python SyntheticDataGenerator/data_generator.py
  .The generated data will be stored in the synthetic_data directory. Agent B will use this data to populate the Chroma vector database.


## Project Structure

multi-agent-tech-support/
├── agents/
│   ├── AgentA.py           # Agent A source code
│   └── AgentB.py           # Agent B source code
├── SyntheticDataGenerator/
│   └── data_generator.py   # Script to create synthetic Q&A data
├── synthetic_data/         # Output data (subfolders for each category)
├── chat_history/           # Folder where chat logs are saved
├── logs/                   # Folder where Agent A & B logs are stored
├── TechAssistDB/           # Chroma vectorstore data
├── app.py                  # Main Streamlit app
├── testAgentB.py           # Script to test Agent B individually
├── testAgents.py           # Script to test Agents A & B together
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (e.g., OPENAI_API_KEY)
└── README.md               # Project documentation (this file)

## Running the Application

1. Test the Agents:
  .Test Agent B:  python testAgentB.py
  .Test Agents A & B together:  python testAgents.py

2. Run the Streamlit App:
   Start the Streamlit-based web application: streamlit run app.py

The application will be available at http://localhost:8501

## Features
Bilingual support (English/French)
Intelligent query routing between agents
RAG-based technical knowledge retrieval
Chat history management
Conversation saving and loading
Professional web interface

## Development Notes and Assumptions
Agent A handles initial user interaction
Uses confidence scoring to determine when to consult Agent B
Automatic escalation for installation and technical queries

## RAG System:
Uses Chroma as vector store
Optimized for technical support documentation
Maintains separate English and French knowledge bases


## Technical Decisions:
Streamlit for rapid deployment and user-friendly interface
Local file storage for chat history and synthetic data
Both agents log their activities in the logs folder. Each day has a separate log file.

## Limitations:
Requires OpenAI API key
English and French support only

## Python Version
The code has been tested with Python 3.9+.

## License
This project is licensed under the MIT License



