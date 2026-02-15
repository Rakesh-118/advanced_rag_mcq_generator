from typing import Dict


class PromptBuilderError(Exception):
    """Custom exception for prompt construction errors."""
    pass


DIFFICULTY_MAP: Dict[str, str] = {
    "Easy": "Focus on direct factual recall and simple definitions.",
    "Medium": "Test conceptual understanding and relationships between ideas.",
    "Hard": "Create analytical, scenario-based, and inference-driven questions with subtle distractors."
}


def build_mcq_prompt(
    content: str,
    num_questions: int,
    difficulty: str,
    bloom_level: str
) -> str:
    """
    Build structured prompt for MCQ generation.

    Args:
        content: Retrieved relevant text content
        num_questions: Number of MCQs to generate
        difficulty: Selected difficulty level
        bloom_level: Selected Bloom taxonomy level

    Returns:
        Formatted prompt string
    """

    if difficulty not in DIFFICULTY_MAP:
        raise PromptBuilderError("Invalid difficulty level provided.")

    if not content or not content.strip():
        raise PromptBuilderError("Content for prompt cannot be empty.")

    difficulty_instruction = DIFFICULTY_MAP[difficulty]

    prompt = f"""
You are an expert academic question designer.

Generate {num_questions} multiple-choice questions strictly from the given content.

Difficulty Level: {difficulty}
Difficulty Description: {difficulty_instruction}

Bloom's Taxonomy Level: {bloom_level}

Rules:
1. Each question must have exactly four options (A, B, C, D).
2. Only one option must be correct.
3. Provide a concise explanation for the correct answer.
4. Avoid repeating concepts.
5. Distractors must be plausible and conceptually close to the correct answer.
6. Do not generate content outside the provided text.

Content:
\"\"\"
{content}
\"\"\"

Return output strictly in this JSON format:

{{
  "mcqs": [
    {{
      "question": "",
      "options": {{
        "A": "",
        "B": "",
        "C": "",
        "D": ""
      }},
      "answer": "",
      "explanation": ""
    }}
  ]
}}
"""

    return prompt.strip()
