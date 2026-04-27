from crewai import Agent, Task, Crew, Process, LLM
from .tools import query_documents
from .config import get_crewai_llm

class EducationCrew:
    def __init__(self):
        self.llm = get_crewai_llm()
        
    def create_agents(self):
        # 1. Summarization Agent
        # Uses prompt engineering in the backstory and goal to guide LLM behavior
        summarizer = Agent(
            role='Senior Document Summarizer',
            goal='Provide clear, concise, and comprehensive summaries of the textbook or reference materials based strictly on the provided documents.',
            backstory=(
                'You are an expert educator who excels at extracting the core concepts and '
                'key points from complex textbooks, making them easy for students to understand. '
                'You never invent information; you only use what is explicitly stated in the source text.'
            ),
            verbose=True,
            allow_delegation=False,
            tools=[query_documents],
            llm=self.llm
        )
        
        # 2. Q&A Agent
        qa_agent = Agent(
            role='Question Answering Specialist',
            goal='Answer user questions accurately based ONLY on the retrieved reference documents.',
            backstory=(
                'You are a meticulous teaching assistant. Your strength lies in finding the exact '
                'answers from the provided textbooks and explaining them clearly. You never guess. '
                'If the answer is not in the text, you clearly state that the information is missing.'
            ),
            verbose=True,
            allow_delegation=False,
            tools=[query_documents],
            llm=self.llm
        )
        
        # 3. MCQ Generator Agent
        mcq_generator = Agent(
            role='Assessment Creator',
            goal='Generate challenging and educational Multiple Choice Questions (MCQs) with correct answers and explanations based on the document text.',
            backstory=(
                'You are a seasoned instructional designer. You know how to create MCQs that test '
                'true comprehension rather than just rote memorization. Every question you create '
                'includes the correct answer and a brief explanation referencing the source material.'
            ),
            verbose=True,
            allow_delegation=False,
            tools=[query_documents],
            llm=self.llm
        )
        
        return summarizer, qa_agent, mcq_generator

    def create_tasks(self, summarizer, qa_agent, mcq_generator, topic, user_question):
        # Prompt Engineering: Detailed instructions in task descriptions
        
        # Task 1: Summarize
        summary_task = Task(
            description=(
                f'Search the documents for the topic: "{topic}". '
                f'Provide a comprehensive summary of the key concepts found. '
                f'Make sure to cover the definitions, main ideas, and any critical examples mentioned in the text.'
            ),
            expected_output='A well-structured summary (3-4 paragraphs) covering the main points of the topic based entirely on the provided documents.',
            agent=summarizer
        )
        
        # Task 2: Answer specific question
        qa_task = Task(
            description=(
                f'Based on the documents related to "{topic}", answer the following specific question: "{user_question}". '
                f'You must search the documents for the specific keywords in the user question. '
                f'If the answer is not in the documents, state clearly: "I could not find the answer to this question in the provided materials."'
            ),
            expected_output='A clear, direct answer to the user\'s question, citing the concepts from the text, or a statement that the answer is not found.',
            agent=qa_agent
        )
        
        # Task 3: Generate MCQs
        mcq_task = Task(
            description=(
                f'Based on the information retrieved about "{topic}", generate 3 Multiple Choice Questions (MCQs). '
                f'Requirements for each MCQ:\n'
                f'1. A clear question testing understanding.\n'
                f'2. 4 options labeled A, B, C, D.\n'
                f'3. Clearly indicate the correct option.\n'
                f'4. Provide a 1-2 sentence explanation of WHY the option is correct based on the text.'
            ),
            expected_output='A formatted list of 3 MCQs with options, the correct answer, and an explanation for each.',
            agent=mcq_generator
        )
        
        return [summary_task, qa_task, mcq_task]

    def run(self, topic, user_question):
        summarizer, qa_agent, mcq_generator = self.create_agents()
        tasks = self.create_tasks(summarizer, qa_agent, mcq_generator, topic, user_question)
        
        # Process.sequential means tasks run one after another. 
        # The output of previous tasks can serve as context for subsequent tasks.
        crew = Crew(
            agents=[summarizer, qa_agent, mcq_generator],
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        return result
