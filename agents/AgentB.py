from typing import List, Dict, Optional, Tuple
import os
import json
from datetime import datetime
import logging
import glob
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import StdOutCallbackHandler
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI

class AgentB:
    def __init__(self, model="gpt-4o-mini", db_name="TechAssistDB"):
        # Set up logging
        self._setup_logger()
        self.logger.info("Initializing Agent B")
        
        # Initialize components
        self.model = model
        self.db_name = db_name
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.5, model_name=self.model)
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        self.vectorstore = None
        self.conversation_chain = None
        
        # Track the last retrieved documents for metadata
        self.last_retrieved_docs = []

    def _setup_logger(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger('AgentB')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        os.makedirs('logs', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/agent_b_{datetime.now().strftime("%Y%m%d")}.log')
        console_handler = logging.StreamHandler()
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def process_json_file(self, json_file: str, doc_type: str) -> List[Document]:
        """Process a single JSON file and convert to documents"""
        self.logger.info(f"Processing file: {json_file}")
        docs = []
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for lang_key in ['EN', 'FR']:
                    if lang_key in data and isinstance(data[lang_key], list):
                        for entry in data[lang_key]:
                            question = entry.get("question", "").strip()
                            answer = entry.get("answer", "").strip()
                            if question and answer:
                                page_content = f"Q: {question}\nA: {answer}"
                                docs.append(Document(
                                    page_content=page_content,
                                    metadata={
                                        "doc_type": doc_type,
                                        "language": lang_key,
                                        "source": json_file
                                    }
                                ))
            return docs
        except Exception as e:
            self.logger.error(f"Error processing file {json_file}: {e}")
            return []

    def initialize_knowledge_base(self, data_folder: str):
        """Initialize the knowledge base from the data folder"""
        self.logger.info(f"Initializing knowledge base from: {data_folder}")
        folders = glob.glob(os.path.join(data_folder, "*"))
        documents = []

        for folder in folders:
            doc_type = os.path.basename(folder)
            json_files = glob.glob(os.path.join(folder, "*.json"))
            for json_file in json_files:
                documents.extend(self.process_json_file(json_file, doc_type))

        self.logger.info(f"Processed {len(documents)} documents")

        if os.path.exists(self.db_name):
            self.logger.info("Removing existing vector store")
            Chroma(persist_directory=self.db_name, embedding_function=self.embeddings).delete_collection()

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_name
        )

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            callbacks=[StdOutCallbackHandler()]
        )

        self.logger.info("Knowledge base initialized successfully")

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query"""
        if not self.vectorstore:
            self.logger.warning("Vector store not initialized")
            return []
            
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            self.last_retrieved_docs = retriever.get_relevant_documents(query)
            self.logger.info(f"Retrieved {len(self.last_retrieved_docs)} relevant documents")
            return self.last_retrieved_docs
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []

    def calculate_response_confidence(self, docs: List[Document]) -> float:
        """Calculate confidence score based on retrieved documents"""
        if not docs:
            return 0.0
        
        # Calculate confidence based on document relevance
        total_score = sum(doc.metadata.get("score", 0.5) for doc in docs)
        avg_score = total_score / len(docs)
        return min(avg_score, 1.0)

    def get_response(self, query: str) -> str:  # Change return type to str
        """Generate a response for the given query"""
        self.logger.info(f"Processing query: {query}")
    
        if not self.conversation_chain:
            self.logger.error("Conversation chain not initialized")
            return "I apologize, but I'm not able to access the technical knowledge base at the moment."

        try:
            result = self.conversation_chain.invoke({
            "question": query
            })
        
            # If result is a dict, extract just the answer
            if isinstance(result, dict):
                return result["answer"]  # Return only the answer
            return result
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your query."
    
    def reset_conversation(self):
        """Reset the conversation memory"""
        self.logger.info("Resetting conversation memory")
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        if self.conversation_chain:
            self.conversation_chain.memory = self.memory