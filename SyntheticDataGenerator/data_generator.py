import os
import json
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import logging
from datetime import datetime

class SyntheticDataGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Setup logging
        self._setup_logger()
        
        # Define categories and templates
        self.categories = {
            'installation': [
                "How do I install {software_name} on {os}?",
                "I'm getting an error '{error_message}' while installing {software_name}.",
            ],
            'configuration': [
                "How can I configure {software_name} to {action}?",
                "What is the best way to set up {software_name} for {use_case}?",
            ],
            'feature_inquiries': [
                "Does {software_name} support {feature}?",
                "How does {software_name} handle {feature}?",
            ],
            'bug_reports': [
                "I'm experiencing {issue} when I {action} in {software_name}.",
                "There's a bug with {feature} in {software_name}; can you help?",
            ],
            'performance_issues': [
                "Why is {software_name} running slow on my {device}?",
                "How can I improve the performance of {software_name} when {action}?",
            ]
        }

    def _setup_logger(self):
        """Set up logging configuration"""
        self.logger = logging.getLogger('DataGenerator')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(
            f'logs/data_generator_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def generate_questions(self, category: str, template: str, software_name: str) -> Dict:
        """Generate Q&A pairs for a given category and template"""
        prompt = (
            f"I am designing a chatbot for a technical support system about {software_name}.\n\n"
            f"Please create 10 example questions and their answers based on the category '{category}'.\n"
            f"The questions should be in French and English, and must test the system's knowledge "
            f"in a realistic way.\n"
            f"Here are some examples for context: {template}\n\n"
            f"Please format the output exactly like this template:\n"
            '''
            {
                "EN": [
                    {
                        "question": "Example English question?",
                        "answer": "Example English answer."
                    }
                ],
                "FR": [
                    {
                        "question": "Example French question?",
                        "answer": "Example French answer."
                    }
                ]
            }
            '''
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that generates Q&A pairs in both English and French. Always output valid JSON that matches the exact template structure provided."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"Error generating questions for category '{category}': {e}")
            return None

    def generate_dataset(self, output_folder: str, software_name: str = "MyApp"):
        """Generate complete dataset with all categories"""
        self.logger.info(f"Starting dataset generation for {software_name}")
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate data for each category
        for category, templates in self.categories.items():
            self.logger.info(f"Generating data for category: {category}")
            
            # Create category folder
            category_folder = os.path.join(output_folder, category)
            os.makedirs(category_folder, exist_ok=True)
            
            # Generate questions for each template
            for i, template in enumerate(templates, 1):
                self.logger.info(f"Processing template {i} for {category}")
                
                generated_data = self.generate_questions(category, template, software_name)
                
                if generated_data:
                    # Validate data structure
                    if not all(key in generated_data for key in ["EN", "FR"]):
                        self.logger.error(f"Invalid data structure for {category} template {i}")
                        continue
                    
                    # Save to file
                    output_file = os.path.join(category_folder, f"questions_{i}.json")
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(generated_data, f, ensure_ascii=False, indent=4)
                    
                    self.logger.info(f"Saved questions to {output_file}")
                else:
                    self.logger.error(f"Failed to generate data for {category} template {i}")

def main():
    # Create generator instance
    generator = SyntheticDataGenerator()
    
    # Generate dataset
    output_folder = "synthetic_data"
    generator.generate_dataset(output_folder)
    
    print("Dataset generation completed. Check the logs for details.")

if __name__ == "__main__":
    main()