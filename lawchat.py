import streamlit as st
import pickle
import os
import faiss
import numpy as np
import re

from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# --------------------------------------------------------------------------
# 0. ✅ 환경 변수 로드
# --------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

GENERAL_API_KEY  = os.getenv("GENERAL_API_KEY")
FINETUNE_API_KEY = os.getenv("FINETUNE_API_KEY")

if GENERAL_API_KEY is None:
    raise ValueError("❌ GENERAL_API_KEY가 .env에서 로드되지 않았습니다.")
if FINETUNE_API_KEY is None:
    raise ValueError("❌ FINETUNE_API_KEY가 .env에서 로드되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = str(GENERAL_API_KEY)

finetune_client = OpenAI(api_key=str(FINETUNE_API_KEY))
general_client  = OpenAI(api_key=str(GENERAL_API_KEY))

FINETUNED_MODEL_ID = "ft:gpt-4.1-mini-2025-04-14:dbdbdeep::CiuSaiDu"

# --------------------------------------------------------------------------
# 1. ✅ 계약서 문서 + 임베딩 로드
# --------------------------------------------------------------------------
@st.cache_resource
def load_docs_and_vectors():
    with open(r"C:\문서\새 폴더\계약_documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(r"C:\문서\새 폴더\계약_embeddings.pkl", "rb") as f:
        vectors = pickle.load(f)
    return docs, vectors

# --------------------------------------------------------------------------
# 2. ✅ FAISS 벡터스토어 생성
# --------------------------------------------------------------------------
@st.cache_resource
def create_vectorstore(_docs, _vectors):
    dim = len(_vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(_vectors).astype("float32"))

    wrapped_docs = [
        Document(page_content=d) if isinstance(d, str) else d
        for d in _docs
    ]

    doc_dict = {str(i): wrapped_docs[i] for i in range(len(wrapped_docs))}
    docstore = InMemoryDocstore(doc_dict)
    index_to_docstore_id = {i: str(i) for i in range(len(wrapped_docs))}

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=str(GENERAL_API_KEY)
    )

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    return vectorstore

# --------------------------------------------------------------------------
# 3. ✅ RAG 체인
# --------------------------------------------------------------------------
@st.cache_resource
def initialize_rag_chain(_vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """당신은 계약서 조항 검색 AI입니다.
다음 문서를 참고하여 정확하게 답하세요.
⚠️ 문서의 파일명이나 식별자는 절대 출력하지 마세요.

{context}"""),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=str(GENERAL_API_KEY)
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# --------------------------------------------------------------------------
# 4. ✅ 파인튜닝 LLM (쉬운 설명)
# --------------------------------------------------------------------------
def explain_with_finetuned_model(clause: str):
    try:
        res = finetune_client.chat.completions.create(
            model=FINETUNED_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 계약서를 쉽게 설명하는 도우미입니다. 반드시 1~4문장으로 핵심만 요약해서 말하세요."
                },
                {
                    "role": "user",
                    "content": clause
                }
            ],
            temperature=0.2
        )

        return res.choices[0].message.content

    except Exception as e:
        print("❌ 파인튜닝 모델 호출 실패:", e)
        return "⚠️ 현재 쉬운 설명 모델이 정상적으로 동작하지 않습니다."

# --------------------------------------------------------------------------
# 5. ✅ 일반 LLM (위험 요소 분석)
# --------------------------------------------------------------------------
def analyze_risk_with_general_llm(clause: str):
    prompt = f"""
다음 계약서 조항에서 근로자에게 불리하거나
주의해야 할 위험 요소를 2~3개 요약하고,
각 항목마다 왜 위험한지도 설명하세요.

{clause}
"""
    res = general_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return res.choices[0].message.content

# --------------------------------------------------------------------------
# 6. ✅ Streamlit 채팅형 UI
# --------------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📄 계약서 이해 AI")
st.caption("계약서 조항 검색 · 쉬운 설명 · 위험 요소 분석을 한 번에!")

# ✅ 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 기존 채팅 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

docs, vectors = load_docs_and_vectors()
vectorstore = create_vectorstore(docs, vectors)
rag_chain = initialize_rag_chain(vectorstore)

# --------------------------------------------------------------------------
# 7. ✅ 사용자 입력 (채팅 방식)
# --------------------------------------------------------------------------
prompt = st.chat_input("계약서 관련 질문을 입력하세요 👇")

if prompt:
    # ✅ 사용자 메시지 저장
    st.session_state.messages.append({
        "role": "human",
        "content": prompt
    })

    with st.chat_message("human"):
        st.markdown(prompt)

    # ✅ AI 응답
    with st.chat_message("ai"):
        with st.spinner("📚 관련 조항 검색 중..."):
            rag_response = rag_chain.invoke({"input": prompt})
            raw_clause = rag_response["answer"]

            # ✅ [파일명.json] 제거
            clause = re.sub(r"\[[^\]]+\.json\]\s*", "", raw_clause)

        st.markdown("### 📌 관련 계약서 조항")
        st.write(clause)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ 쉬운 설명")
            easy = explain_with_finetuned_model(clause)
            st.write(easy)

        with col2:
            st.markdown("### ⚠️ 위험 요소 분석")
            risk = analyze_risk_with_general_llm(clause)
            st.write(risk)

        # ✅ AI 전체 응답을 하나의 말풍선으로 저장
        full_ai_message = f"""
### 📌 관련 계약서 조항
{clause}

---

### ✅ 쉬운 설명
{easy}

---

### ⚠️ 위험 요소 분석
{risk}
"""
        st.session_state.messages.append({
            "role": "ai",
            "content": full_ai_message
        })
