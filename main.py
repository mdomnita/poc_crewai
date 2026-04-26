import os
from src.agents import EducationCrew

def main():
    print("=====================================================")
    print(" Educational Document Insight Assistant (Multi-Agent)")
    print("=====================================================\n")
    
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' directory.")
        
    print("Please ensure you have placed your PDF and CSV files in the 'data' directory.")
    print("Also, ensure Ollama is running locally with the necessary models pulled (e.g., 'llama3' and 'nomic-embed-text').\n")
    
    from dotenv import load_dotenv
    load_dotenv()
    topic = input("Enter a topic you want to learn about from the documents (e.g., 'Machine Learning'): ")
    if not topic.strip():
        print("Topic cannot be empty. Exiting.")
        return
        
    question = input(f"Enter a specific question you have about '{topic}': ")
    
    print("\nStarting the Agentic Workflow... This may take a few minutes as the agents process the data.\n")
    
    try:
        crew = EducationCrew()
        result = crew.run(topic=topic, user_question=question)
        
        print("\n" + "="*50)
        print("FINAL EDUCATIONAL REPORT")
        print("="*50)
        print(result)
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    main()
