from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings

from core.schema import MCQ


class DeduplicationError(Exception):
    """Custom exception for deduplication failures."""
    pass


def remove_similar_mcqs(
    mcqs: List[MCQ],
    similarity_threshold: float = 0.85
) -> List[MCQ]:
    """
    Remove semantically similar MCQs using embedding cosine similarity.

    Args:
        mcqs: List of validated MCQ objects
        similarity_threshold: Cosine similarity threshold

    Returns:
        List of unique MCQs
    """

    try:
        if not mcqs:
            return []

        embeddings_model = OpenAIEmbeddings()

        unique_mcqs = []
        stored_vectors = []

        for mcq in mcqs:
            vector = embeddings_model.embed_query(mcq.question)

            if not stored_vectors:
                unique_mcqs.append(mcq)
                stored_vectors.append(vector)
                continue

            similarities = cosine_similarity(
                [vector],
                stored_vectors
            )[0]

            max_similarity = np.max(similarities)

            if max_similarity < similarity_threshold:
                unique_mcqs.append(mcq)
                stored_vectors.append(vector)

        return unique_mcqs

    except Exception as e:
        raise DeduplicationError(f"Failed during MCQ deduplication: {str(e)}")

