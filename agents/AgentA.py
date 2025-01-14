from typing import Dict, Tuple, Optional, List
import json
from openai import OpenAI
import logging
from datetime import datetime

class AgentA:
    """
    First-line support agent that handles initial user queries and determines
    when to escalate to Agent B for technical expertise.
    """
    
    def __init__(self, name: str = "Agent A", model: str = "gpt-3.5-turbo", agent_b=None):
        """
        Initialize Agent A with configuration and connection to Agent B.
        
        Args:
            name (str): Name identifier for the agent
            model (str): OpenAI model to use
            agent_b: Reference to Agent B instance for escalation
        """
        self.name = name
        self.model = model
        self.client = OpenAI()
        self.agent_b = agent_b
        self.conversation_history: List[Dict] = []
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(f"Initialized {self.name} with model {self.model}")

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for Agent A"""
        logger = logging.getLogger(f"{self.name}")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler(f'logs/agent_a_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def _analyze_query(self, query: str) -> Dict:
        """
        Analyze the user query to determine its nature and requirements.
        
        Args:
            query (str): User's input query
            
        Returns:
            Dict: Analysis results including complexity, category, and escalation needs
        """
        analysis_prompt = f"""Analyze this technical support query and provide structured information:
        Query: {query}
        
        Provide analysis in JSON format with the following structure:
        {{
            "complexity": "low/medium/high",
            "category": "installation/configuration/bug/performance/security/other",
            "needs_technical_expertise": boolean,
            "confidence": float between 0 and 1,
            "keywords": [relevant technical terms],
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"Error in query analysis: {e}")
            return {
                "complexity": "high",
                "needs_technical_expertise": True,
                "confidence": 0.0,
                "reasoning": "Error in analysis"
            }

    def _format_response(self, content: str, analysis: Dict) -> str:
        """Format the response based on analysis and content"""
        if analysis.get("needs_technical_expertise", False):
            return (
                f"I understand your question about {', '.join(analysis.get('keywords', []))}. "
                f"This seems to be a {analysis.get('category', 'technical')} issue that requires "
                f"specific expertise. Let me consult with our technical expert.\n\n{content}"
            )
        return content

    def _should_escalate(self, analysis: Dict) -> bool:
        """
        Determine if the query should be escalated to Agent B based on analysis.
        
        Args:
            analysis (Dict): Query analysis results
            
        Returns:
            bool: True if query should be escalated, False otherwise
        """
        conditions = [
            analysis.get("complexity") == "high",
            analysis.get("needs_technical_expertise", False),
            analysis.get("confidence", 1.0) < 0.7
        ]
        return any(conditions)

    def _get_direct_response(self, query: str, analysis: Dict) -> str:
        """
        Generate a direct response for queries that don't need escalation.
        
        Args:
            query (str): User's query
            analysis (Dict): Query analysis results
            
        Returns:
            str: Generated response
        """
        prompt = f"""You are {self.name}, a helpful technical support agent.
        Respond to this user query: "{query}"
        
        Category: {analysis.get('category')}
        Complexity: {analysis.get('complexity')}
        
        Provide a clear, helpful response in natural language.
        If you're not sure about something, be honest about it.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating direct response: {e}")
            return "I apologize, but I'm having trouble generating a response. Please try again."

    def process_query(self, query: str) -> str:
        """
        Process a user query and return appropriate response.
        
        Args:
            query (str): User's input query
            
        Returns:
            str: Response to the user
        """
        self.logger.info(f"Processing query: {query}")
        
        # Analyze the query
        analysis = self._analyze_query(query)
        self.logger.info(f"Query analysis: {analysis}")
        
        # Determine if escalation is needed
        if self._should_escalate(analysis) and self.agent_b is not None:
            self.logger.info("Escalating to Agent B")
            try:
                response = self.agent_b.get_response(query)
            except Exception as e:
                self.logger.error(f"Error getting response from Agent B: {e}")
                response = "I apologize, but I'm having trouble getting the technical information you need. Please try again."
        else:
            self.logger.info("Handling query directly")
            response = self._get_direct_response(query, analysis)
        
        # Format and store response
        formatted_response = self._format_response(response, analysis)
        self.conversation_history.append({
            "query": query,
            "response": formatted_response,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        })
        
        return formatted_response

    def get_conversation_history(self) -> List[Dict]:
        """Return the conversation history"""
        return self.conversation_history