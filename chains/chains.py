from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def create_rag_chain(vectorstore):
    llm = ChatOllama(model='qwen2.5', temperature=0)
    
    base_retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 10})
    reranker_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_kwargs={'device': 'cpu'} 
    )
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    contextualize_q_system_prompt = (
        "Dựa trên lịch sử trò chuyện và câu hỏi mới nhất của người dùng, "
        "hãy viết lại một câu hỏi độc lập để tối ưu hóa việc tìm kiếm trong tài liệu. "
        "YÊU CẦU QUAN TRỌNG: Giữ nguyên ý nghĩa tiếng Việt nhưng HÃY BỔ SUNG thêm các "
        "thuật ngữ chuyên ngành bằng Tiếng Anh (nếu có) vào câu hỏi. "
        "KHÔNG trả lời câu hỏi, CHỈ trả về đúng 1 câu hỏi đã được viết lại."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    qa_system_prompt = (
        "Bạn là một trợ lý AI nội bộ nghiêm ngặt. \n"
        "QUY TẮC:\n"
        "1. CHỈ sử dụng thông tin từ 'Context' để trả lời.\n"
        "2. LUÔN trả lời bằng Tiếng Việt rõ ràng, tự nhiên.\n"
        "3. Nếu không có trong 'Context', hãy nói: 'Tôi không tìm thấy thông tin này'.\n"
        "4. KHÔNG đoán, KHÔNG bịa đặt, KHÔNG dùng kiến thức bên ngoài.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain