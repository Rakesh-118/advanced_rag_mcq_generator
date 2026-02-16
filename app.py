import streamlit as st

from core.loaders import load_pdf, load_docx, validate_text_input, DocumentLoaderError
from core.vectorstore import create_vector_store, retrieve_relevant_chunks, VectorStoreError
from core.prompt import build_mcq_prompt, PromptBuilderError
from core.generator import generate_mcqs_from_prompt, parse_and_validate_mcqs, LLMGenerationError
from core.deduplicator import remove_similar_mcqs, DeduplicationError


st.set_page_config(page_title="Advanced RAG MCQ Generator", layout="wide")

st.title("🧠 Advanced RAG-Based MCQ Generator")


# ----------------------
# SESSION STATE INIT
# ----------------------

if "quiz_mcqs" not in st.session_state:
    st.session_state.quiz_mcqs = []

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "score" not in st.session_state:
    st.session_state.score = 0

if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = False


# ----------------------
# INPUT SECTION
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
# GENERATE MCQs
# ----------------------

if st.button("Generate MCQs"):

    try:
        # Load input
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                text = load_pdf(uploaded_file)
            else:
                text = load_docx(uploaded_file)
        else:
            text = validate_text_input(input_text)

        st.success("Document loaded successfully.")

        with st.spinner("Generating unique MCQs..."):

            vector_store = create_vector_store(text)

            retrieved_chunks = retrieve_relevant_chunks(
                vector_store,
                query="Generate high-quality MCQs",
                k=5
            )

            combined_content = "\n".join(retrieved_chunks)

            prompt = build_mcq_prompt(
                content=combined_content,
                num_questions=num_questions,
                difficulty=difficulty,
                bloom_level=bloom_level
            )

            raw_response = generate_mcqs_from_prompt(prompt, temperature=0.7)

            validated_output = parse_and_validate_mcqs(raw_response)

            unique_mcqs = remove_similar_mcqs(validated_output.mcqs)

        # Store MCQs in session
        st.session_state.quiz_mcqs = unique_mcqs
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.quiz_mode = False

        st.success(f"{len(unique_mcqs)} Unique MCQs Generated Successfully!")

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


# ----------------------
# DISPLAY GENERATED MCQs
# ----------------------

if st.session_state.quiz_mcqs and not st.session_state.quiz_mode:

    st.markdown("## Generated MCQs")

    for idx, mcq in enumerate(st.session_state.quiz_mcqs, 1):
        st.markdown(f"### Q{idx}: {mcq.question}")

        for key, value in mcq.options.items():
            st.write(f"{key}. {value}")

        st.success(f"Answer: {mcq.answer}")
        st.info(f"Explanation: {mcq.explanation}")
        st.markdown("---")

    st.markdown("### Do you want to take a quiz?")

    if st.button("Start Quiz"):
        st.session_state.quiz_mode = True
        st.rerun()


# ----------------------
# QUIZ MODE SECTION
# ----------------------


if st.session_state.quiz_mode and st.session_state.quiz_mcqs:

    mcqs = st.session_state.quiz_mcqs
    index = st.session_state.current_question

    if "answered" not in st.session_state:
        st.session_state.answered = False

    if index < len(mcqs):

        mcq = mcqs[index]

        st.markdown(f"## Question {index + 1} of {len(mcqs)}")
        st.markdown(f"### {mcq.question}")

        selected = st.radio(
            "Choose your answer:",
            list(mcq.options.keys()),
            key=f"radio_{index}",
            format_func=lambda x: f"{x}. {mcq.options[x]}"
        )

        # SUBMIT BUTTON
        if not st.session_state.answered:
            if st.button("Submit Answer", key=f"submit_{index}"):

                if selected == mcq.answer:
                    st.success("Correct! 🎉")
                    st.session_state.score += 1
                else:
                    st.error(f"Wrong! Correct answer: {mcq.answer}")

                st.info(f"Explanation: {mcq.explanation}")

                st.session_state.answered = True

        # NEXT BUTTON
        if st.session_state.answered:
            if st.button("Next Question", key=f"next_{index}"):

                st.session_state.current_question += 1
                st.session_state.answered = False
                st.rerun()

    else:
        st.markdown("## Quiz Completed! 🎯")
        st.markdown(f"### Final Score: {st.session_state.score} / {len(mcqs)}")

        if st.button("Restart Quiz"):

            st.session_state.quiz_mode = False
            st.session_state.quiz_mcqs = []
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.session_state.answered = False

            st.rerun()
