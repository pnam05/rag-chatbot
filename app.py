import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Technical RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Be Vietnam Pro', sans-serif; }

    .chat-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        color: #fff;
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(30,58,95,.35);
    }
    .msg-user {
        background: #1e3a5f; color: #e8f4fd;
        padding: .75rem 1rem; border-radius: 18px 18px 4px 18px;
        margin: .4rem 0 .4rem 15%; box-shadow: 0 2px 8px rgba(0,0,0,.15);
    }
    .msg-assistant {
        background: #f0f4f8; color: #1a2e44;
        padding: .75rem 1rem; border-radius: 18px 18px 18px 4px;
        margin: .4rem 15% .4rem 0; box-shadow: 0 2px 8px rgba(0,0,0,.08);
    }
    .source-box {
        font-size: 0.85rem; color: #555;
        background: #fafafa; border: 1px solid #eee;
        padding: 0.5rem; border-radius: 8px; margin-top: 0.5rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="⚙️ Đang kết nối não bộ AI...")
def load_chain():
    load_dotenv()
    from database.database import get_or_update_vectorstore
    from chains.chains import create_rag_chain
    vectorstore = get_or_update_vectorstore()
    return create_rag_chain(vectorstore)

if "messages" not in st.session_state:
    st.session_state.messages = []          
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []        

with st.sidebar:
    st.markdown("## ⚙️ Cài đặt")
    
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("<small>Model: **Qwen 2.5**<br>Reranker: **MiniLM-L6**</small>", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <div style="font-size:2rem; margin-right:15px">🤖</div>
    <div>
        <h1 style="margin:0; font-size:1.4rem">Technical RAG Assistant</h1>
        <p style="margin:0; font-size:0.85rem; opacity:0.8">Hỏi đáp dựa trên tài liệu nội bộ</p>
    </div>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    role_label = "Bạn" if msg["role"] == "user" else "Trợ lý"
    align = "right" if msg["role"] == "user" else "left"
    css_class = "msg-user" if msg["role"] == "user" else "msg-assistant"
    
    st.markdown(f'<div style="text-align:{align}; font-size:.7rem; opacity:.5">{role_label}</div>'
                f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)
    
    if "sources" in msg and msg["sources"]:
        with st.expander("🔍 Nguồn tham khảo"):
            for s in msg["sources"]:
                st.markdown(f"📍 **{s['file']}** - Trang {s['page']}")

user_input = st.chat_input("Nhập câu hỏi kỹ thuật của bạn…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_msg = st.session_state.messages[-1]["content"]
    
    rag_chain = load_chain()
    full_answer = ""
    source_docs = []

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        
        stream_handler = rag_chain.stream({
            "input": last_user_msg,
            "chat_history": st.session_state.lc_history,
        })

        for chunk in stream_handler:
            if "context" in chunk:
                source_docs = chunk["context"]
            
            if "answer" in chunk:
                full_answer += chunk["answer"]
                answer_placeholder.markdown(full_answer + "▌")

        answer_placeholder.markdown(full_answer)
        
        formatted_sources = []
        if source_docs:
            with st.expander("🔍 Nguồn tham khảo"):
                unique_sources = set()
                for doc in source_docs:
                    f_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    page = doc.metadata.get('page', 0) + 1 
                    unique_sources.add((f_name, page))
                
                for f_name, page in sorted(unique_sources):
                    st.markdown(f"📍 **{f_name}** — Trang {page}")
                    formatted_sources.append({"file": f_name, "page": page})

    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_answer,
        "sources": formatted_sources
    })
    st.session_state.lc_history.extend([
        HumanMessage(content=last_user_msg),
        AIMessage(content=full_answer),
    ])

    if len(st.session_state.lc_history) > 10:
        st.session_state.lc_history = st.session_state.lc_history[-10:]