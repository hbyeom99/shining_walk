
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LangChain imports ---
from langchain_community.document_loaders import PyPDFLoader
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


# =============================================================================
# 2. 데이터 구조 및 유틸리티 함수 (크롤링, 데이터 준비)
# =============================================================================
BASE_URL = "https://tour.gwangju.go.kr"
HOME_URL = f"{BASE_URL}/home/main.cs"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.1 Safari/537.36"
)

@dataclass
class CrawlItem:
    title: str
    link: str
    summary: str
    lang: str

def get_soup(url: str) -> Optional[BeautifulSoup]:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        if r.status_code != 200:
            print(f"Error fetching {url}: Status Code {r.status_code}")
            return None
        return BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        print(f"Exception fetching {url}: {e}")
        return None

def absolutize(href: str) -> str:
    if href.startswith("http"):
        return href
    if href.startswith("/"):
        return BASE_URL + href
    return BASE_URL + "/" + href.lstrip("/")

SECTION_HREF_PATTERNS = ["/sub.cs?m=", "/eng/", "/tour/info/"]
# Using raw string for regex to avoid issues with backslashes
DETAIL_PATTERNS = [re.compile(r"/tour/info/[^?]+\.cs\?act=view&.*infoId=\d+")]

def extract_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
        tag.extract()
    summary_candidates = []
    for cls in ["summary", "desc", "text", "txt", "cont", "con_txt", "info", "basic", "bbs_view", "view_cont", "body_wrap"]:
        for el in soup.select(f".{cls}"):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 50:
                summary_candidates.append(txt)
    for p in soup.select("div.contents p, div.body p, main p"):
        txt = p.get_text(" ", strip=True)
        if txt and len(txt) > 50:
            summary_candidates.append(txt)

    if summary_candidates:
        best = max(summary_candidates, key=len)
    else:
        body_text = soup.body.get_text(" ", strip=True) if soup.body else ""
        best = body_text

    # Using raw string for regex
    return re.sub(r"\s+", " ", best)[:1500]

def detect_lang(text: str) -> str:
    return "ko" if re.search(r"[\u3131-\uD7A3]", text) else "en"

def crawl_home_and_details(start_url: str, max_pages: int = 140, delay_sec: float = 0.5) -> List[CrawlItem]:
    items: List[CrawlItem] = []
    to_crawl_queue = deque([start_url])
    processed_urls: set[str] = set([start_url])

    page_count = 0

    print(f"크롤링 시작: {start_url}")

    pbar = tqdm(total=max_pages, desc="크롤링 진행", unit="페이지")

    while to_crawl_queue and page_count < max_pages:
        current_url = to_crawl_queue.popleft()
        pbar.set_description(f"크롤링 중: {current_url[:60]}...")

        soup = get_soup(current_url)
        if not soup:
            pbar.write(f"  > 페이지 가져오기 실패: {current_url}")
            continue

        page_count += 1
        pbar.update(1)

        if any(p.search(current_url) for p in DETAIL_PATTERNS):
            title = None
            for sel in ["h1", "h2", ".tit", ".title", ".page_tit", ".bbs_tit", "meta[property='og:title']"]:
                el = soup.select_one(sel)
                if el:
                    title = el.get("content") if el.name == "meta" else el.get_text(strip=True)
                    break
            if not title:
                title = soup.title.get_text(strip=True) if soup.title else current_url
            summary = extract_text(soup)
            lang = detect_lang(summary)
            if len(summary) > 50:
                 items.append(CrawlItem(title=title, link=current_url, summary=summary, lang=lang))
                 pbar.write(f"  > 상세 페이지 데이터 추출 완료: {title[:50]}...")


        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            absolute_url = absolutize(href)

            if not absolute_url.startswith(BASE_URL): continue
            if "#" in absolute_url: absolute_url = absolute_url.split("#")[0]
            if absolute_url.endswith((".zip", ".pdf", ".doc", ".xls", ".ppt", ".hwp", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".avi", ".wmv")): continue
            if "javascript:" in absolute_url: continue
            if "mailto:" in absolute_url: continue

            is_section_link = any(p in absolute_url for p in SECTION_HREF_PATTERNS)
            is_detail_link = any(p.search(absolute_url) for p in DETAIL_PATTERNS)
            is_pagination_link = "page=" in absolute_url

            if (is_section_link or is_detail_link or is_pagination_link) and absolute_url not in processed_urls:
                 to_crawl_queue.append(absolute_url)
                 processed_urls.add(absolute_url)


        time.sleep(delay_sec)

    pbar.close()
    print(f"크롤링 완료. 총 {len(items)}건의 데이터 추출.")
    print(f"처리된 URL 수: {len(processed_urls)}")
    return items


def items_to_documents(items: List[CrawlItem]) -> List[Document]:
    docs = []
    for it in items:
        meta = {"source": it.link, "title": it.title, "lang": it.lang}
        content = it.summary if isinstance(it.summary, str) else str(it.summary)
        docs.append(Document(page_content=content, metadata=meta))
    return docs

def make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, separators=["\n\n", "\n", ". ", ".", " "])

def load_pdf_urls(pdf_urls: List[str]) -> List[Document]:
    loaded: List[Document] = []
    cache_dir = ".cache_pdfs"
    os.makedirs(cache_dir, exist_ok=True)
    for url in tqdm(pdf_urls, desc="PDF 다운로드 및 로딩"):
        try:
            fn = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest() + ".pdf")
            if not os.path.exists(fn):
                r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True)
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(fn, "wb") as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1], leave=False
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            if os.path.exists(fn):
                loader = PyPDFLoader(fn)
                try:
                    pdf_docs = loader.load()
                    loaded.extend(pdf_docs)
                except Exception as pdf_e:
                    print(f"Error loading PDF {url}: {pdf_e}")
        except Exception as e:
            print(f"Error downloading or processing PDF {url}: {e}")
            continue
    return loaded

# =============================================================================
# 3. 벡터 스토어 구축 및 RAG 체인 설정
# =============================================================================

# Global variables to maintain state
crawled_items: List[CrawlItem] = []
vector_store: Optional[FAISS] = None
session_store: Dict[str, BaseChatMessageHistory] = {}

def build_faiss(docs: List[Document]) -> FAISS:
    if not OPENAI_API_KEY:
        # This check should ideally happen before calling build_faiss
        # but is included here as a safeguard.
        raise ValueError("OpenAI API key not set. Cannot build embeddings.")

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    return FAISS.from_documents(docs, embeddings)

def build_qa_chain(model_name: str = "gpt-4o-mini"):
    if not OPENAI_API_KEY:
         # This check should ideally happen before calling build_qa_chain
         # but is included here as a safeguard.
         raise ValueError("OpenAI API key not set. Cannot build QA chain.")

    # Using raw string and explicit newlines for system prompt
    system_prompt = (
r"""당신은 이제 광주 관광에 대한 깊은 지식과 열정을 가진 최고의 여행 전문가입니다.\n너는 광주광역시 공식 관광 포털의 콘텐츠와 추가 PDF를 요약/검색해주는\n관광 도우미 역할을 수행하며, 사용자 질문 의도를 정확히 파악하고,\n마치 친한 여행 가이드처럼 친절하고 상세하게 설명해야 합니다.\n제공된 정보를 바탕으로 다양한 관점에서 답변해줘, 이전에 언급한 내용은 반복하지 않도록 노력해줘\n비슷한 내용을 물어본다면 기존에 답했던 내용은 중복되었다는 뜻이니 제외하고 다른 정보를 검색해서 알려줘\n없는 내용은 절대로 지어내서 알려주면 안되며 모르는 내용은 관련된 주소를 링크해서 대신 대답해줘\n\n딱딱한 정보 나열보다는 매력적인 여행 이야기를 들려주듯 답변해주세요.\n광주의 숨겨진 명소, 맛집, 행사, 교통 정보 등 여행객에게 꼭 필요한\n실질적인 정보를 중심으로 제공해주세요. 사용자의 여행 스타일과 예산을\n고려하여 맞춤형 추천을 덧붙여주면 더욱 좋습니다.\n\n답변은 한국어로 하고, 최종에 정보의 출처 링크를 bullet으로 명확하게 포함하여\n신뢰도를 높여주세요."""
    )

    template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "사용자 프로필: {profile}\\n\\n질문: {question}\\n\\n참고 컨텍스트: {context}\\n\\n요청: 컨텍스트 기반으로 정확히 답하고, 부족하면 일반 상식은 피하고 '공식 포털에서 정보를 찾지 못했다'라고 말해. 한국어로 답하기. 최종에 출처를 bullet로 정리."),
    ])

    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name, temperature=0.2)

    def format_docs(docs: List[Document]) -> str:
        chunks = []
        added_sources = set()
        for i, d in enumerate(docs):
            t = d.metadata.get("title", f"문서 {i+1}")
            s = d.page_content[:500]
            src = d.metadata.get("source", "출처 불명")
            chunks.append(f"- **{t}** ([출처]({src}))\\n{s}...") # Use \\n for literal newline in f-string
            added_sources.add(src)

        sources_list = "\\n\\n**참고 자료:**\\n" + "\\n".join([f"- {s}" for s in added_sources]) if added_sources else ""

        return "\\n\\n".join(chunks) + sources_list


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

store: Dict[str, BaseChatMessageHistory] = {}
def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# =============================================================================
# 4. Gradio 앱 로직 함수 (자동 빌드 및 챗봇 처리)
# =============================================================================

def crawl_data_for_autobuild() -> str:
    """Performs crawling for the autobuild process."""
    global crawled_items
    print("크롤링 함수 시작...")
    try:
        crawled_items = crawl_home_and_details(start_url=HOME_URL, max_pages=140, delay_sec=0.2)
        uniq: Dict[str, CrawlItem] = {}
        for it in crawled_items:
            uniq[it.link] = it
        crawled_items = list(uniq.values())

        if crawled_items:
            print(f"크롤링 완료: 총 {len(crawled_items)}건의 데이터를 수집했습니다.")
            return f"크롤링 완료: 총 {len(crawled_items)}건의 데이터를 수집했습니다."
        else:
            print("크롤링된 데이터가 없습니다.")
            return "크롤링된 데이터가 없습니다."
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"크롤링 중 오류 발생: {e}")
        return f"크롤링 중 오류 발생: {e}"

def build_vectorstore_for_autobuild() -> str:
    """Builds vectorstore for the autobuild process."""
    global crawled_items, vector_store, OPENAI_API_KEY

    print("벡터DB 구축 함수 시작...")

    if not OPENAI_API_KEY:
         print("OpenAI API 키가 설정되지 않았습니다. 벡터DB 구축 건너뜁니다.")
         # Modified message to guide user on Hugging Face Secrets
         return "🔑 OpenAI API 키가 설정되지 않아 벡터DB 구축을 건너뜁니다. Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요."


    if not crawled_items:
        print("크롤링된 데이터가 없습니다. 벡터DB 구축 건너뜁니다.")
        return "크롤링된 데이터가 없습니다. 벡터DB 구축 건너뜁니다."

    docs = items_to_documents(crawled_items)

    # Example PDF URLs (add actual URLs if needed)
    # pdf_urls = ["http://example.com/some_gwangju_guide.pdf"]
    # if pdf_urls:
    #      print(f"추가 PDF {len(pdf_urls)}건 로딩 중...")
    #      pdf_docs = load_pdf_urls(pdf_urls)
    #      docs.extend(pdf_docs)
    #      print(f"PDF 로딩 완료. 총 {len(pdf_docs)}건 추가.")


    if not docs:
        print("문서가 비어 있습니다. 벡터DB 구축 건너뜁니다.")
        return "문서가 비어 있습니다. 벡터DB 구축 건너뜁니다."

    try:
        splitter = make_splitter()
        chunks = splitter.split_documents(docs)
        if not chunks:
             print("텍스트 분할 결과가 없습니다. 벡터DB 구축 건너뜁니다.")
             return "텍스트 분할 결과가 없습니다. 벡터DB 구축 건너뜁니다."

        print(f"텍스트 청크 생성 완료: 총 {len(chunks)}개")

        print("벡터 임베딩 생성 및 스토어 구축 중...")
        vector_store = build_faiss(chunks) # build_faiss will check for API key internally
        print("벡터DB 생성/갱신 완료.")
        return f"벡터DB 생성/갱신 완료: 총 {len(chunks)}개의 텍스트 청크."
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"벡터DB 구축 중 오류 발생: {e}")
        return f"벡터DB 구축 중 오류 발생: {e}"

def build_vectorstore_with_progress(progress=gr.Progress()):
    """Orchestrates the autobuild process with progress reporting."""
    global OPENAI_API_KEY

    if not OPENAI_API_KEY:
         print("API 키 없음: 데이터 준비 건너뜁니다.")
         # Modified message to guide user on Hugging Face Secrets
         return "🔑 OpenAI API 키가 설정되지 않아 데이터 준비를 건너뜁니다. Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요."


    steps = [
        "데이터 크롤링 중...",
        "문서 변환 및 분할 중...",
        "벡터 임베딩 생성 및 스토어 구축 중...",
        "마무리 중..."
    ]

    progress(0, desc=steps[0])
    crawl_result_msg = crawl_data_for_autobuild()
    time.sleep(0.5)

    progress(1/len(steps), desc=steps[1])
    time.sleep(0.5)

    progress(2/len(steps), desc=steps[2])
    build_result_msg = build_vectorstore_for_autobuild() # build_vectorstore_for_autobuild handles API key check
    time.sleep(0.5)

    progress(3/len(steps), desc=steps[3])
    time.sleep(0.5)

    final_message = f"✅ 준비 완료! {crawl_result_msg}, {build_result_msg}"
    print(final_message)
    return final_message


def rag_chatbot_for_autobuild(user_message: str, chat_history: list, user_profile: str) -> str:
    """Handles chatbot interaction with RAG and memory for the autobuild process (Retrieval separated)."""
    global vector_store, OPENAI_API_KEY, session_store

    if not OPENAI_API_KEY:
        print("API 키 없음: 챗봇 응답 불가")
        # Modified message to guide user on Hugging Face Secrets
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
                gr.Image(
                    value="고양이3.gif",
                    visible=True,
                    elem_id="loading_gif",
                )
                # Modified initial message to guide user on Hugging Face Secrets
                loading_status = gr.Textbox(label="진행 상태", interactive=False, show_label=False, value="시작 준비 중... Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요.")


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
                    outputs=user_profile_state
                )

            def user_chat(user_message, history, current_profile):
                if not user_message:
                    return history, ""

                response = rag_chatbot_for_autobuild(user_message, history, current_profile)

                # Ensure history format is compatible with type='messages'
                # The response is a string, need to format it as a message dictionary
                new_history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": response}]

                return new_history, ""

            msg.submit(
                user_chat,
                inputs=[msg, chatbot, user_profile_state],
                outputs=[chatbot, msg]
            )
            clear.click(lambda: [], inputs=None, outputs=chatbot)

            # --- OpenAI API Key Input (Removed as key is handled by Secrets) ---
            # gr.Markdown("### 🔑 OpenAI API Key 설정")
            # openai_key_input = gr.Textbox(...)
            # def set_openai_key(key): ...
            # openai_key_input.change(...)


        # ---- (C) 실행 시 자동 DB 구축 트리거 ----
        def init_build(progress=gr.Progress()):
            print("init_build function started by demo.load")
            # The build_vectorstore_with_progress function now includes API key check
            status_message = build_vectorstore_with_progress(progress)
            print(f"init_build finished with status: {status_message}")
            # Only switch to chatbot screen if build was successful (vector_store is not None)
            if vector_store is not None:
                return gr.update(visible=False), gr.update(visible=True), status_message
            else:
                # Stay on loading screen if build failed (e.g., missing API key)
                return gr.update(visible=True), gr.update(visible=False), status_message


        demo.load(
            init_build,
            inputs=None,
            outputs=[loading_screen, chatbot_screen, loading_status]
        )

    return demo

# =============================================================================
# 6. 실행
# =============================================================================
if __name__ == "__main__":
    print("앱 실행 시작...")

    try:
        import gradio
        gradio.close_all()
        print("기존 Gradio 인스턴스 종료 시도.")
    except Exception as e:
        print(f"Gradio 인스턴스 종료 중 오류 발생 (무시 가능): {e}")

    # OPENAI_API_KEY is now loaded globally at the top from os.getenv

    if not OPENAI_API_KEY:
        print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. 앱이 시작되지만 벡터DB 구축 및 챗봇 기능이 제한됩니다. Hugging Face Secrets에 'OPENAI_API_KEY'를 설정해주세요.")
    else:
        print("OPENAI_API_KEY 환경 변수 감지됨.")

    demo = gradio_ui()
    print("Gradio UI 실행 중...")
    # Removed share=True for default Hugging Face Spaces deployment
    demo.launch(debug=True)
