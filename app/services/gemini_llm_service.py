# app/services/gemini_llm_service.py

from typing import List
from app.config import settings
from app.db.models import LLMEvaluation, Message

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --- 1. Define the LLM and Output Structure ---

# We define the Pydantic model *again* for LangChain's parser
# This ensures the parser knows exactly what to ask the LLM for.
class LLMEvaluationOutput(BaseModel):
    evaluation: str = Field(description="AI's feedback on the student's last answer. Be concise and constructive.")
    new_question: str = Field(description="AI's next question for the student. The question should be relevant to the topic and the last answer.")

# Initialize the LLM
# We use .with_structured_output to make JSON parsing reliable
llm = ChatGoogleGenerativeAI(
    # Use a stable model identifier available in your environment
    model="gemini-2.5-flash",
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.7
).with_structured_output(LLMEvaluationOutput)

# Initialize the JSON parser (kept for clarity)
parser = JsonOutputParser(pydantic_object=LLMEvaluationOutput)

# --- 2. Prompt Templates ---

FIRST_QUESTION_PROMPT = """
You are an expert AI viva examiner.
Your role is to conduct a verbal viva (oral exam) for a student.
The topic of the viva is: {topic}
The student's class level is: {class_level}

Start the viva by introducing yourself briefly and asking the very first question.
Keep your response concise.

Provide your response ONLY in the following JSON format.
{format_instructions}
"""

EVALUATION_PROMPT = """
You are an expert AI viva examiner conducting a verbal viva.
The viva topic is: {topic}
The student's class level is: {class_level}

You are in the middle of the viva. Here is the transcript so far:
{history}

The student just gave this answer to your last question:
Student Answer: "{student_answer}"

Your task is to:
1.  Briefly evaluate the student's answer. (e.g., "That's correct...", "Good point, but consider...")
2.  Ask the next logical question to continue the viva.

Provide your response ONLY in the following JSON format.
{format_instructions}
"""

# --- 3. Helper Function to Format History ---

def format_history(history: List[Message]) -> str:
    """Converts the list of Message objects into a string for the LLM."""
    if not history:
        return "No history yet."
    
    formatted = []
    for msg in history:
        if msg.speaker == "ai":
            formatted.append(f"AI Examiner: {msg.text}")
        else:
            formatted.append(f"Student: {msg.text}")
    return "\n".join(formatted)

# --- 4. Main Service Functions (Called by Orchestrator) ---

async def get_ai_first_question(topic: str, class_level: str) -> LLMEvaluation:
    """
    Gets the initial question from the LLM.
    """
    print(f"--- LLM Service: Getting first question for topic: {topic} ---")
    
    prompt_template = ChatPromptTemplate.from_template(
        template=FIRST_QUESTION_PROMPT,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt_template | llm
    
    # We use .ainvoke() for an async call
    response_model: LLMEvaluationOutput = await chain.ainvoke({
        "topic": topic,
        "class_level": class_level
    })
    
    # In the first question, there is no "evaluation", so we provide a short placeholder.
    response_model.evaluation = "Let's begin."
    
    print(f"--- LLM Service: Got first question: {response_model.new_question[:60]}... ---")
    
    # Convert from LangChain's Pydantic model to our app's Pydantic model
    # IMPORTANT: convert to a plain dict first to satisfy model_validate's input expectations.
    try:
        response_dict = response_model.model_dump()
        return LLMEvaluation.model_validate(response_dict)
    except Exception as e:
        # Provide clear logs for debugging if validation fails
        print(f"--- LLM Service: Error validating first question response! {e} ---")
        raise

async def get_ai_evaluation(
    topic: str, 
    class_level: str, 
    history: List[Message], 
    student_answer: str
) -> LLMEvaluation:
    """
    Gets an evaluation and the next question from the LLM.
    """
    print(f"--- LLM Service: Evaluating answer: {student_answer[:60]}... ---")
    
    prompt_template = ChatPromptTemplate.from_template(
        template=EVALUATION_PROMPT,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = prompt_template | llm
    
    formatted_history = format_history(history)
    
    response_model: LLMEvaluationOutput = await chain.ainvoke({
        "topic": topic,
        "class_level": class_level,
        "history": formatted_history,
        "student_answer": student_answer
    })
    
    print(f"--- LLM Service: Got evaluation: {response_model.evaluation[:60]}... ---")
    print(f"--- LLM Service: Got next question: {response_model.new_question[:60]}... ---")

    # Convert from LangChain's Pydantic model to our app's Pydantic model
    try:
        response_dict = response_model.model_dump()
        return LLMEvaluation.model_validate(response_dict)
    except Exception as e:
        print(f"--- LLM Service: Error validating evaluation response! {e} ---")
        raise