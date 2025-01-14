import os
from dotenv import load_dotenv
from agents.AgentA import AgentA
from agents.AgentB import AgentB

# Load environment variables
load_dotenv()

def test_agents():
    # Initialize Agent B first (since Agent A needs it)
    print("Initializing Agent B...")
    agent_b = AgentB(model="gpt-4o-mini", db_name="TechAssistDB")
    
    # Initialize the knowledge base
    print("Initializing knowledge base...")
    agent_b.initialize_knowledge_base("synthetic_data")
    
    # Initialize Agent A with Agent B
    print("Initializing Agent A...")
    agent_a = AgentA(agent_b=agent_b)
    
    # Test question
    #test_query = "How do I install MyApp on Windows?"
    test_query = "I'm experiencing frequent crashes and high CPU usage when running MyApp on Windows 11. The error log shows dependency conflicts, and it happens especially when I try to process large files. I've already tried restarting the application but the problem persists. Can you help me diagnose and fix this?"
    print(f"\nTesting with query: {test_query}")
    
    # Get response
    response = agent_a.process_query(test_query)
    print("\nResponse:", response)

if __name__ == "__main__":
    test_agents()