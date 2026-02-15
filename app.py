import streamlit as st

from core.loaders import load_pdf, load_docx, validate_text_input, DocumentLoaderError
from core.vectorstore import create_vector_store, retrieve_relevant_chunks, VectorStoreError
from core.prompt import build_mcq_prompt, PromptBuilderError
from core.generator import generate_mcqs_from_prompt, parse_and_validate_mcqs, LLMGenerationError
from core.deduplicator import remove_similar_mcqs, DeduplicationError


st.set_page_config(page_title="Advanced RAG MCQ Generator", layout="wide")

st.title("🧠 Advanced RAG-Based MCQ Generator")


# ----------------------
# Input Section
# ----------------------

input_text = st.text_area("Enter Text (Optional)")

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

num_questions = st.slider("Number of Questions", 1, 20, 5)

difficulty = st.selectbox("Select Difficulty", ["Easy", "Medium", "Hard"])

bloom_level = st.selectbox(
    "Bloom's Taxonomy Level",
    ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
)


# ----------------------
# Generation Button
# ----------------------

if st.button("Generate MCQs"):

    try:
        # ----------------------
        # Load Input
        # ----------------------

        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                text = load_pdf(uploaded_file)
            else:
                text = load_docx(uploaded_file)
        else:
            text = validate_text_input(input_text)

        st.info("Document loaded successfully.")

        # ----------------------
        # Create Vector Store
        # ----------------------

        vector_store = create_vector_store(text)

        # ----------------------
        # Retrieve Relevant Content
        # ----------------------

        retrieved_chunks = retrieve_relevant_chunks(
            vector_store,
            query="Generate high-quality MCQs",
            k=5
        )

        combined_content = "\n".join(retrieved_chunks)

        # ----------------------
        # Build Prompt
        # ----------------------

        prompt = build_mcq_prompt(
            content=combined_content,
            num_questions=num_questions,
            difficulty=difficulty,
            bloom_level=bloom_level
        )

        # ----------------------
        # Generate MCQs
        # ----------------------

        raw_response = generate_mcqs_from_prompt(
            prompt,
            temperature=0.7
        )

        # ----------------------
        # Parse + Validate
        # ----------------------

        validated_output = parse_and_validate_mcqs(raw_response)

        # ----------------------
        # Deduplicate
        # ----------------------

        unique_mcqs = remove_similar_mcqs(validated_output.mcqs)

        # ----------------------
        # Display Results
        # ----------------------

        st.success(f"{len(unique_mcqs)} Unique MCQs Generated")

        for idx, mcq in enumerate(unique_mcqs, 1):
            st.markdown(f"### Q{idx}: {mcq.question}")

            for key, value in mcq.options.items():
                st.write(f"{key}. {value}")

            st.success(f"Answer: {mcq.answer}")
            st.info(f"Explanation: {mcq.explanation}")
            st.markdown("---")

    except (
        DocumentLoaderError,
        VectorStoreError,
        PromptBuilderError,
        LLMGenerationError,
        DeduplicationError,
        ValueError
    ) as e:
        st.error(str(e))

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

