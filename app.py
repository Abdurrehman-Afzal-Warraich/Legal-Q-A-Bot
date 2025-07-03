import streamlit as st
from src.pdf_utils import extract_text_from_pdf, split_text_into_chunks
from src.embedding_utils import generate_embeddings
from src.chroma_utils import create_collection, query_collection
from src.qa_utils import get_answer_from_context, summarize_context

st.set_page_config(page_title="Legal Document QA Bot")
st.title("📄 Legal Document QA Bot")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "collection" not in st.session_state:
    st.session_state.collection = None

# Handle query param cleanup
if st.query_params.get("clear_input"):
    st.query_params.clear()

# File uploader
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type="pdf")

if uploaded_file is not None:
    uploaded_file_path = "temp.pdf"
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.read())

    pdf_text = extract_text_from_pdf(uploaded_file_path)
    text_chunks = split_text_into_chunks(pdf_text)
    embeddings = generate_embeddings(text_chunks)

    collection = create_collection(name="legal_docs", persist_path="data/chroma_db")
    collection.add(
        documents=text_chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(text_chunks))]
    )

    st.session_state.collection = collection
    st.success("✅ PDF uploaded and indexed!")

# Display chat history
for question, answer in st.session_state.chat_history:
    st.markdown(f"**You:** {question}")
    st.markdown(f"**Bot:** {answer}")

# Input + ask button
user_question = st.text_input("Ask another legal question:", key="question_input")
ask_button = st.button("Ask", disabled=not user_question.strip())

# When user clicks "Ask"
if ask_button and user_question:
    if st.session_state.collection is not None:
        question_embedding = generate_embeddings([user_question])[0]
        results = query_collection(st.session_state.collection, question_embedding, n_results=3)
        top_contexts = " ".join(results["documents"][0])

        if user_question.lower().startswith("summarize"):
            summary = summarize_context(top_contexts)
            st.session_state.chat_history.append((user_question, summary["answer"]))
        else:
            answer = get_answer_from_context(user_question, top_contexts)
            st.session_state.chat_history.append((user_question, answer["answer"]))

        st.query_params["clear_input"] = "true"
        st.rerun()
    else:
        st.warning("⚠️ Please upload a PDF first to ask questions.")

# Fallback message
if uploaded_file is None and not st.session_state.chat_history:
    st.info("⬆ Please upload a PDF document to get started.")
