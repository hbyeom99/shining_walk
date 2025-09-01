# 필요한 라이브러리 명시적 설치 (이미 설치되어 있다면 이 셀을 건너뛰거나 주석 처리하세요)
!pip install -q langchain-openai
!pip install -q langchain-community
!pip install -q pypdf beautifulsoup4 requests tqdm gradio # Include pypdf here

import os
import gradio as gr
import time
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import deque

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- Secure key load ---
# OPENAI_API_KEY is loaded from Hugging Face Secrets or environment variables
# In Colab for testing, attempt to load from secrets or define manually
try:
    from google.colab import userdata
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
         print("WARNING: 'OPENAI_API_KEY' not found in Colab secrets. Please add it or define it manually.")
except Exception as e:
    print(f"WARNING: Could not load OPENAI_API_KEY from Colab secrets: {e}")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Fallback to env var


# --- LangChain imports ---
# Make sure these are installed:
# !pip install -q langchain-community langchain-openai langchain-text-splitters faiss-cpu
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Import FAISS module explicitly for loading
from langchain_community.vectorstores.faiss import FAISS as FAISS_Vectorstore

# Import pypdf explicitly as it's used by PyPDFLoader internally
import pypdf


# =============================================================================
# 2. 데이터 구조 및 유틸리티 함수 (데이터 준비)
# =============================================================================

# Crawling and PDF loading functions are removed or commented out
# BASE_URL = "https://tour.gwangju.go.kr" (...)
# def crawl_home_and_details(...)
# def load_data_from_pdf(...)

def make_splitter() -> RecursiveCharacterTextSplitter:
    # Splitter is still needed if you rebuild from source or process new docs,
    # but not strictly for loading a saved index. Keep for consistency maybe.
    return RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ". ", ".", " "])


# =============================================================================
# 3. 벡터 스토어 구축 및 RAG 체인 설정
# =============================================================================

# Global variables to maintain state
vector_store: Optional[FAISS] = None # Will be loaded from file
session_store: Dict[str, BaseChatMessageHistory] = {}

# build_faiss function is not needed if loading from saved index
# def build_faiss(docs: List[Document]) -> FAISS: (...)

def build_qa_chain(model_name: str = "gpt-4o-mini"):
    global OPENAI_API_KEY
    if not OPENAI_API_KEY:
         raise ValueError("OpenAI API key not set. Cannot build QA chain.")

    # Using raw string and explicit newlines for system prompt
    system_prompt = (
'''당신은 이제 광주 관광에 대한 깊은 지식과 열정을 가진 최고의 여행 전문가입니다.
너는 광주광역시 공식 관광 포털의 콘텐츠와 추가 PDF를 요약/검색해주는
관광 도우미 역할을 수행하며, 사용자 질문 의도를 정확히 파악하고,
마치 친한 여행 가이드처럼 친절하고 상세하게 설명해야 합니다.
제공된 정보를 바탕으로 다양한 관점에서 답변해줘, 이전에 언급한 내용은 반복하지 않도록 노력해줘
비슷한 내용을 물어본다면 기존에 답했던 내용은 중복되었다는 뜻이니 제외하고 다른 정보를 검색해서 알려줘
없는 내용은 절대로 지어내서 알려주면 안되며 모르는 내용은 관련된 주소를 링크해서 대신 대답해줘

딱딱한 정보 나열보다는 매력적인 여행 이야기를 들려주듯 답변해주세요.
광주의 숨겨진 명소, 맛집, 행사, 교통 정보 등 여행객에게 꼭 필요한
실질적인 정보를 중심으로 제공해주세요. 사용자의 여행 스타일과 예산을
고려하여 맞춤형 추천을 덧붙여주면 더욱 좋습니다.

답변은 한국어로 하고, 최종에 정보의 출처 링크를 bullet으로 명확하게 포함하여
신뢰도를 높여주세요.'''
    )

    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "사용자 프로필: {profile}\n\n질문: {question}\n\n참고 컨텍스트: {context}\n\n요청: 컨텍스트 기반으로 정확히 답하고, 부족하면 일반 상식은 피하고 '공식 포털에서 정보를 찾지 못했다'라고 말해. 한국어로 답하기. 최종에 출처를 bullet로 정리."),
    ])

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name, temperature=0.2)

    def format_docs(docs: List[Document]) -> str:
        chunks = []
        added_sources = set()
        for i, d in enumerate(docs):
            t = d.metadata.get("title", f"문서 {i+1}")
            s = d.page_content[:500]
            src = d.metadata.get("source", "출처 불명") # 'source' should now contain original URL from crawled data
            chunks.append(f"- **{t}** ([출처]({src}))\n{s}...") # Use \n for literal newline in f-string
            added_sources.add(src)

        sources_list = "\n\n**참고 자료:**\n" + "\n".join([f"- {s}" for s in added_sources]) if added_sources else ""

        return "\n\n".join(chunks) + sources_list


    qa_chain = (
         RunnableParallel({
             "context": lambda x: format_docs(x["context"]),
             "question": RunnablePassthrough(),
             "profile": RunnablePassthrough()
         })
        | template
        | llm
        | StrOutputParser()
    )

    return qa_chain

# Session history store
store: Dict[str, BaseChatMessageHistory] = {}
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# =============================================================================
# 4. Gradio 앱 로직 함수 (자동 빌드/로딩 및 챗봇 처리)
# =============================================================================

# build_vectorstore_for_autobuild is replaced by load_vectorstore
# def build_vectorstore_for_autobuild() -> str: (...)

def load_vectorstore_from_saved() -> str:
    """Loads the vectorstore from the saved directory."""
    global vector_store, OPENAI_API_KEY

    print("벡터DB 로드 함수 시작...")

    if not OPENAI_API_KEY:
         print("OpenAI API 키가 설정되지 않았습니다. 벡터DB 로드 건너뜁니다.")
         return "🔑 OpenAI API 키가 설정되지 않아 벡터DB 로드를 건너뜁니다. Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요."

    save_dir = "faiss_index"
    if not os.path.exists(save_dir) or not os.path.exists(os.path.join(save_dir, "index.faiss")) or not os.path.exists(os.path.join(save_dir, "index.pkl")):
        print(f"경고: 저장된 벡터 스토어 디렉토리 '{save_dir}'을(를) 찾을 수 없거나 파일이 완전하지 않습니다. 벡터DB 로드 건너뜁니다.")
        return f"⚠️ 저장된 벡터 스토어 디렉토리 '{save_dir}'을(를) 찾을 수 없습니다. 먼저 벡터 스토어를 구축하고 저장해주세요."

    try:
        print(f"'{save_dir}'에서 벡터 스토어 로드 중...")
        # Need embeddings object to load FAISS
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vector_store = FAISS_Vectorstore.load_local(save_dir, embeddings, allow_dangerous_deserialization=True)
        print("벡터DB 로드 완료.")
        return f"✅ 벡터DB 로드 완료: '{save_dir}'에서 로드되었습니다."
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"벡터DB 로드 중 오류 발생: {e}")
        # Include API key check in error message if it's the likely cause
        if "AuthenticationError" in str(e) or "api_key" in str(e).lower():
             return "🔑 OpenAI API 키 인증 오류가 발생했습니다. Hugging Face Secrets에 설정된 키를 확인해주세요."
        else:
             return f"벡터DB 로드 중 오류 발생: {e}"


def build_or_load_vectorstore_with_progress(progress=gr.Progress()):
    """Orchestrates the vectorstore loading process with progress reporting."""
    global OPENAI_API_KEY

    if not OPENAI_API_KEY:
         print("API 키 없음: 데이터 준비 건너뜁니다.")
         return "🔑 OpenAI API 키가 설정되지 않아 데이터 준비를 건너뜁니다. Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요."


    steps = [
        "저장된 벡터DB 로드 중...", # Updated step description
        "마무리 중..."
    ]

    progress(0, desc=steps[0])
    status_message = load_vectorstore_from_saved() # Call the loading function
    time.sleep(0.5)

    progress(1/len(steps), desc=steps[1])
    time.sleep(0.5)

    final_message = f"✅ 준비 완료! {status_message}" # Updated final message
    print(final_message)
    return final_message


def rag_chatbot_for_autobuild(user_message: str, chat_history: list, user_profile: str) -> str:
    """Handles chatbot interaction with RAG and memory."""
    global vector_store, OPENAI_API_KEY, session_store

    if not OPENAI_API_KEY:
        print("API 키 없음: 챗봇 응답 불가")
        return "🔑 OpenAI API 키가 설정되지 않았습니다. Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요."


    if vector_store is None:
        print("벡터DB 준비 안됨: 챗봇 응답 불가")
        return "⏳ 데이터 준비 중입니다. 잠시 후 다시 시도해주세요."

    session_id = "gradio_session_auto"
    history = get_history(session_id)

    try:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 10, "lambda_mult": 0.5}
        )

        retrieved_docs = retriever.get_relevant_documents(str(user_message))

        qa_chain = build_qa_chain() # build_qa_chain will check for API key internally

        conversational_qa_chain = RunnableWithMessageHistory(
             qa_chain,
             get_history,
             input_messages_key="question",
             history_messages_key="history",
        )

        qa_input = {
             "question": str(user_message),
             "context": retrieved_docs,
             "profile": user_profile
        }

        answer = conversational_qa_chain.invoke(
             qa_input,
             config={"configurable": {"session_id": session_id}},
        )

        return answer

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"챗봇 응답 생성 중 오류 발생: {e}")
        # Include API key check in error message if it's the likely cause
        if "AuthenticationError" in str(e) or "api_key" in str(e).lower():
             return "🔑 OpenAI API 키 인증 오류가 발생했습니다. Hugging Face Secrets에 설정된 키를 확인해주세요."
        else:
             return f"챗bot 응답 생성 중 오류 발생: {e}"


# =============================================================================
# 5. Gradio UI 정의
# =============================================================================

def gradio_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🏞️ 광주관광 친구")

        user_profile_state = gr.State("여행스타일:자유여행, 동행:친구, 예산:보통")

        # ---- (A) 로딩 화면 ----
        with gr.Row(visible=True) as loading_screen:
            with gr.Column():
                gr.Markdown(
                    "### ⏳ 산책을 위해 준비운동 중입니다. 잠시만 기다려주세요! 🐰🌸"
                )
                # Using a public URL for the image instead of a local file
                gr.Image(
                    value="https://cdn.pixabay.com/animation/2025/07/25/00/29/00-29-46-321_512.gif", # Updated image URL
                    visible=True,
                    elem_id="loading_gif",
                )
                # Modified initial message to indicate vectorstore loading
                loading_status = gr.Textbox(label="진행 상태", interactive=False, show_label=False, value="시작 준비 중... 저장된 벡터DB 로드 중...")


        # ---- (B) 챗봇 화면 (초기에는 숨김) ----
        with gr.Column(visible=False) as chatbot_screen:
            gr.Markdown("### 🤖 광주 관광 가이드 챗봇")
            chatbot = gr.Chatbot(label="챗봇", type='messages')
            msg = gr.Textbox(placeholder="광주 관광에 대해 물어보세요!", show_label=False)
            clear = gr.Button("대화 초기화")

            # --- User Profile Input ---
            with gr.Accordion("사용자 프로필 설정", open=False):
                profile_style = gr.Dropdown(label="여행 스타일", choices=["자유여행", "단체여행", "반려동물 동반", "뚜벅이여행"], value="자유여행")
                profile_companion = gr.Dropdown(label="동행", choices=["혼자", "친구", "가족", "연인"], value="친구")
                profile_budget = gr.Dropdown(label="예산", choices=["상관없음", "저렴", "보통", "여유"], value="보통")
                profile_update_btn = gr.Button("프로필 업데이트")

                def update_profile_state(style, companion, budget):
                    profile_str = f"여행스타일:{style}, 동행:{companion}, 예산:{budget}"
                    print(f"사용자 프로필 업데이트: {profile_str}")
                    return profile_str

                profile_update_btn.click(
                    update_profile_state,
                    inputs=[profile_style, profile_companion, profile_budget],
                    outputs=[user_profile_state]
                )

            def user_chat(user_message: str, chat_history: list, current_profile: str) -> Tuple[list, str]:
                if not user_message:
                    return chat_history, ""

                response = rag_chatbot_for_autobuild(user_message, chat_history, current_profile)

                # Ensure history format is compatible with type='messages'
                # The response is a string, need to format it as a message dictionary
                new_history = chat_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": response}]

                return new_history, ""

            msg.submit(
                user_chat,
                inputs=[msg, chatbot, user_profile_state],
                outputs=[chatbot, msg]
            )
            clear.click(lambda: [], inputs=None, outputs=chatbot)


        # ---- (C) 실행 시 자동 DB 로드 트리거 ----
        def init_load(progress=gr.Progress()):
            print("init_load function started by demo.load")
            # The load_vectorstore_from_saved function now handles API key and file checks
            status_message = load_vectorstore_from_saved()
            print(f"init_load finished with status: {status_message}")
            # Only switch to chatbot screen if loading was successful (vector_store is not None)
            if vector_store is not None:
                return gr.update(visible=False), gr.update(visible=True), status_message
            else:
                # Stay on loading screen if loading failed (e.g., missing API key or index)
                return gr.update(visible=True), gr.update(visible=False), status_message


        # Use demo.load() to trigger init_load when the Gradio app starts
        demo.load(
            init_load,
            inputs=None,
            outputs=[loading_screen, chatbot_screen, loading_status]
        )

    return demo

# =============================================================================
# 6. 실행
# =============================================================================
# This section is included for direct execution within a single Colab cell
if __name__ == "__main__":
    print("앱 실행 시작...")

    # Ensure necessary libraries are installed for this single cell execution
    try:
        import gradio
        import langchain_community # Check if common libraries are available
        import langchain_openai
        import faiss # faiss-cpu is the package name, but faiss is the import
        import bs4 # Check for beautifulsoup4 import name
        import requests
        import tqdm
        import pypdf # Add pypdf to the check
    except ImportError:
        print("Installing necessary libraries...")
        # Include pypdf and beautifulsoup4 in the installation list
        !pip install -q langchain-community langchain-openai langchain-text-splitters faiss-cpu pypdf beautifulsoup4 requests tqdm gradio

    # Define OPENAI_API_KEY here for Colab testing if not already defined
    # Replace with your actual key or load from secrets if preferred for testing
    # Make sure the 'faiss_index' directory exists and contains index.faiss and index.pkl
    try:
        # Attempt to load from Colab secrets first
        from google.colab import userdata
        OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
        if not OPENAI_API_KEY:
             print("WARNING: 'OPENAI_API_KEY' not found in Colab secrets. Please add it or define it manually.")
    except Exception as e:
        print(f"WARNING: Could not load OPENAI_API_KEY from Colab secrets: {e}")
        # Fallback to environment variable if running outside Colab secrets context
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


    # Check for API key again before launching demo if needed for loading vectorstore
    if not OPENAI_API_KEY:
         print("\n주의: OpenAI API 키가 설정되지 않았습니다. Gradio 앱은 시작되지만, 벡터DB 로드 및 챗봇 기능은 작동하지 않습니다. Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요.")
         # You might want to skip launching the demo here if API key is essential
         # Or let the load_vectorstore_from_saved function handle the message

    demo = gradio_ui()
    print("Gradio UI 실행 중...")
    # Removed share=True for default Hugging Face Spaces deployment
    # You might need share=True if you want a public link outside Colab
    demo.launch(debug=True)
]

app_py_code = "\n".join(app_py_code_lines)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_py_code)