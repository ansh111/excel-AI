import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document


# Load API key

load_dotenv()

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai.api_key = os.getenv("OPENAI_API_KEY")



st.set_page_config(page_title="Excel Summarizer & Q&A", layout="wide")


st.title("📊 Excel Sheet Summarizer + AI Q&A")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    selected_sheet = st.selectbox("Choose a sheet", sheet_names)

    df = xls.parse(selected_sheet)
    st.write("### Preview", df.head())

    # Step 1: Convert dataframe to text
    def df_to_text(df):
        return "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist())

    raw_text = df_to_text(df)

    # Step 2: Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = splitter.create_documents([raw_text])

    # Step 3: Vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)

    # Step 4: RAG-based QA
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    if st.button("🧠 Summarize Sheet"):
        with st.spinner("Summarizing with OpenAI..."):
        # Take top 3 chunks to fit token limits
         sample_text = "\n\n".join([doc.page_content for doc in texts[:3]])

         summary_prompt = f"""
You are an expert data analyst. Summarize the key patterns, metrics, and observations from the following Excel sheet content:

{sample_text}

Provide a concise, bullet-pointed summary in plain English.
"""

         response = openai.ChatCompletion.create(model="gpt-4",
         messages=[
             {"role": "system", "content": "You are a helpful assistant that summarizes tabular data."},
             {"role": "user", "content": summary_prompt}
         ])
        summary = response.choices[0].message.content
        st.markdown("### 📋 Summary")
        st.markdown(summary)


    st.markdown("### ❓ Ask a Question")
    question = st.text_input("What do you want to know about the sheet?")

    if question:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": question})
            st.markdown("### 💬 Answer")
            st.write(result['result'])
            with st.expander("🔎 Sources"):
                for doc in result['source_documents']:
                    st.write(doc.page_content)