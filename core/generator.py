import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class LLMGenerationError(Exception):
    """Custom exception for LLM generation failures."""
    pass


# Load environment variables
load_dotenv()


def generate_mcqs_from_prompt(prompt: str, temperature: float = 0.7) -> str:
    """
    Generate MCQs using OpenAI Chat model.

    Args:
        prompt: Fully constructed prompt string
        temperature: Controls randomness (higher = more creative)

    Returns:
        Raw LLM response (string)

    Raises:
        LLMGenerationError: If API call fails
    """

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMGenerationError("OpenAI API key not found in environment variables.")

        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1"

        )

        response = llm.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content.strip()

    except Exception as e:
        raise LLMGenerationError(f"MCQ generation failed: {str(e)}")


#JSON PARSING

import json
from pydantic import ValidationError
from core.schema import MCQList


def parse_and_validate_mcqs(raw_response: str) -> MCQList:
    """
    Parse raw LLM JSON output and validate using Pydantic schema.

    Args:
        raw_response: Raw string from LLM

    Returns:
        Validated MCQList object

    Raises:
        LLMGenerationError: If parsing or validation fails
    """
    try:
        data = json.loads(raw_response)
        validated = MCQList(**data)
        return validated

    except json.JSONDecodeError:
        raise LLMGenerationError("LLM returned invalid JSON format.")

    except ValidationError as ve:
        raise LLMGenerationError(f"MCQ schema validation failed: {ve}")
