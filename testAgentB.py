import os
from dotenv import load_dotenv
from agents.AgentB import AgentB

# Load environment variables
load_dotenv()

def test_agent_b():
    # Initialize Agent B
    print("\n=== Initializing Agent B... ===")
    agent_b = AgentB(model="gpt-4o-mini", db_name="TechAssistDB")
    
    # Initialize the knowledge base
    print("\n=== Loading Knowledge Base... ===")
    agent_b.initialize_knowledge_base("synthetic_data")
    
    # Test query
    test_query = "How do I install MyApp on Windows?"
    print(f"\n=== Testing Query: {test_query} ===")
    
    # Test get_response method
    print("\nTesting Agent B Response:")
    response = agent_b.get_response(test_query)
    print("Response:", response)

if __name__ == "__main__":
    test_agent_b()