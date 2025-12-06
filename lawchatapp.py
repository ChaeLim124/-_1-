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
    with open(r"C:\Users\cheri\OneDrive\문서\새 폴더\계약_documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(r"C:\Users\cheri\OneDrive\문서\새 폴더\계약_embeddings.pkl", "rb") as f:
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
# --------------------------------------------------------------------------
# 6. ✅ Streamlit 채팅형 UI (커스텀 레이아웃)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="계약서 이해 AI",
    page_icon="📄",
    layout="wide",
)

# 💄 전체 스타일
st.markdown(
    """
<style>
/* 전체 배경 */
body {
    background: #edf2f7;
}

/* Streamlit 기본 여백 조금 줄이기 */
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* 페이지 중앙 카드 */
.app-bg {
    display: flex;
    justify-content: center;
}

.app-frame {
    width: 100%;
    max-width: 960px;
    background: #ffffff;
    border-radius: 18px;
    box-shadow: 0 12px 35px rgba(15, 35, 95, 0.12);
    padding: 24px 28px 28px;
}

/* 상단 제목 영역 */
.header-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

.menu-icon {
    width: 30px;
    height: 30px;
    border-radius: 999px;
    border: 1px solid #d4e0f4;
    display: flex;
    align-items: center;
    justify-content: center;
}

.menu-icon-bar {
    width: 14px;
    height: 2px;
    border-radius: 999px;
    background: #4b6bb6;
    position: relative;
}
.menu-icon-bar::before,
.menu-icon-bar::after {
    content: "";
    position: absolute;
    width: 14px;
    height: 2px;
    border-radius: 999px;
    background: #4b6bb6;
    left: 0;
}
.menu-icon-bar::before { top: -4px; }
.menu-icon-bar::after  { top:  4px; }

.app-title {
    font-size: 26px;
    font-weight: 750;
    color: #1f2a4d;
}

.app-subtitle {
    font-size: 13px;
    color: #7a8aad;
}

/* 채팅 카드 */
.chat-window {
    margin-top: 16px;
    background: #ffffff;
    border-radius: 18px;
    padding: 16px 20px 18px;
    box-shadow: 0 6px 18px rgba(15, 35, 95, 0.08);
    border: 1px solid #e0e9fb;
}

/* 카톡처럼 가운데 축 */
.chat-inner {
    max-width: 720px;
    margin: 0 auto;
}

/* 한 줄(행) */
.chat-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin-bottom: 12px;
}
.chat-row.user {
    justify-content: flex-end;
}

/* 아바타 */
.avatar {
    width: 28px;
    height: 28px;
    border-radius: 999px;
    background: #e9f1ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    color: #4b7cf5;
}
.avatar.bot {
    background: #edf1f9;
    color: #7b8baa;
}

/* 기본 말풍선 */
.bubble {
    border-radius: 18px;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.4;
    max-width: 420px;
    word-break: keep-all;
}
.bubble.user {
    background: #2f80ff;
    color: #ffffff;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    background: #ffffff;
    border: 1px solid #dfe7f5;
    color: #1f2937;
    border-bottom-left-radius: 4px;
}

/* 최초 인사 카드 */
.bot-card {
    max-width: 520px;
    border-radius: 16px;
    border: 1px solid #dde5f2;
    background: #ffffff;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.bot-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: #34415f;
    padding-bottom: 6px;
    border-bottom: 1px solid #edf1f9;
}
.bot-card-avatar {
    width: 22px;
    height: 22px;
    border-radius: 999px;
    border: 1px solid #d1ddf5;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
}
.bot-card-body {
    font-size: 14px;
    line-height: 1.5;
    color: #2f3a54;
}

/* RAG 결과 카드 */
.answer-card {
    max-width: 520px;
    border-radius: 16px;
    border: 1px solid #dfe7f5;
    background: #ffffff;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.5;
}
.answer-section {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed #e4e9f5;
}
.answer-section:first-child {
    margin-top: 0;
    padding-top: 0;
    border-top: none;
}
.answer-section-title {
    font-weight: 600;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 4px;
    color: #1f2a4d;
}
.answer-section-body {
    font-size: 13.5px;
    color: #374151;
}

/* 입력창도 가운데 정렬 */
.stChatInput {
    margin-top: 1.2rem;
}
.stChatInput > div {
    max-width: 960px;
    margin: 0 auto;
}
</style>
""",
    unsafe_allow_html=True,
)

# ✅ 채팅 기록 초기화 (처음에 인사 메시지 1개)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "ai",
            "type": "intro",
            "content": (
                "안녕하세요, 계약서 이해 AI입니다.\n"
                "궁금한 계약서 조항이나 키워드를 아래 입력창에 적어 주세요.\n"
                '예: "근로시간 조항 설명해줘"'
            ),
        }
    ]

# --------------------------------------------------------------------------
# 7. ✅ 사용자 입력 처리 (st.chat_input)
# --------------------------------------------------------------------------
docs, vectors = load_docs_and_vectors()
vectorstore = create_vectorstore(docs, vectors)
rag_chain = initialize_rag_chain(vectorstore)

user_query = st.chat_input("메시지를 입력하세요")

if user_query:
    # 1) 사용자 메시지 저장
    st.session_state.messages.append(
        {"role": "human", "type": "user", "content": user_query}
    )

    # 2) RAG 검색 + 파인튜닝 + 위험분석
    rag_response = rag_chain.invoke({"input": user_query})
    raw_clause = rag_response["answer"]

    # [파일명.json] 제거
    clause = re.sub(r"\[[^\]]+\.json\]\s*", "", raw_clause)

    easy = explain_with_finetuned_model(clause)
    risk = analyze_risk_with_general_llm(clause)

    # 3) AI 답변 구조화해서 저장
    st.session_state.messages.append(
        {
            "role": "ai",
            "type": "answer",
            "query": user_query,
            "clause": clause,
            "easy": easy,
            "risk": risk,
        }
    )

# --------------------------------------------------------------------------
# 8. ✅ 레이아웃 렌더링 (단순: 웹페이지 위에 채팅만)
# --------------------------------------------------------------------------

# 상단 프레임 열기
st.markdown('<div class="chat-inner">', unsafe_allow_html=True)
# 헤더
st.markdown(
    """
<div class="header-row">
  <div class="menu-icon">
    <div class="menu-icon-bar"></div>
  </div>
  <div>
    <div class="app-title">계약서 이해 AI</div>
    <div class="app-subtitle">계약서 조항 검색 · 쉬운 설명 · 위험 요소 분석</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# 채팅 카드 시작

# 메시지들 렌더링
for msg in st.session_state.messages:
    if msg["role"] == "human":
        # 사용자(파란 말풍선)
        st.markdown(
            f"""
<div class="chat-row user">
  <div class="bubble user">{msg['content']}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        # 최초 인사 카드
        if msg["type"] == "intro":
            body = msg["content"].replace("\n", "<br />")
            st.markdown(
                f"""
<div class="chat-row">
  <div class="avatar bot">👤</div>
  <div class="bot-card">
    <div class="bot-card-header">
      <div class="bot-card-avatar">🤖</div>
      <div>계약서 이해 도우미</div>
    </div>
    <div class="bot-card-body">
      {body}
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        # 답변 카드
        elif msg["type"] == "answer":
            clause_html = msg["clause"].replace("\n", "<br />")
            easy_html = msg["easy"].replace("\n", "<br />")
            risk_html = msg["risk"].replace("\n", "<br />")

            st.markdown(
                f"""
<div class="chat-row">
  <div class="avatar bot">👤</div>
  <div class="answer-card">
    <div class="answer-section">
      <div class="answer-section-title">🔵 관련 계약서 조항</div>
      <div class="answer-section-body">{clause_html}</div>
    </div>
    <div class="answer-section">
      <div class="answer-section-title">✨ 쉬운 설명</div>
      <div class="answer-section-body">{easy_html}</div>
    </div>
    <div class="answer-section">
      <div class="answer-section-title">⚠️ 위험 요소 요약</div>
      <div class="answer-section-body">{risk_html}</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

# 채팅 카드 / 프레임 닫기
st.markdown("</div></div></div></div>", unsafe_allow_html=True)

