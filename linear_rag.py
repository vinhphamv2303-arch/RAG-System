import json
import time
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import TextIteratorStreamer
from threading import Thread

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
PROJECT_ROOT = "/workspace/RAG-System"
GRAPH_PATH = os.path.join(PROJECT_ROOT, "metadata/linear_tri_graph.json")
VECTOR_PATH = os.path.join(PROJECT_ROOT, "metadata/entity_vectors.npy")
KEYS_PATH = os.path.join(PROJECT_ROOT, "metadata/entity_vectors_keys.json")

EMBEDDING_MODEL = "keepitreal/vietnamese-sbert" 
LLM_MODEL = "Qwen/Qwen2.5-14B-Instruct"

TOP_K_ENTITIES = 5  
TOP_K_CHUNKS = 15    

print("\n" + "="*50)
print("🚀 ĐANG KHỞI ĐỘNG HỆ THỐNG LINEARRAG...")
print("="*50)

# ==========================================
# 1. LOAD MODEL & DATA 
# ==========================================
embedder = SentenceTransformer(EMBEDDING_MODEL)

with open(GRAPH_PATH, 'r', encoding='utf-8') as f: tri_graph = json.load(f)
with open(KEYS_PATH, 'r', encoding='utf-8') as f: graph_entities = json.load(f)
entity_embeddings = np.load(VECTOR_PATH)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL, torch_dtype="auto", device_map="auto"
)

print("\n✅ Hệ thống đã sẵn sàng phục vụ!")

# ==========================================
# 2. CÁC HÀM XỬ LÝ LÕI
# ==========================================
def extract_query_entities(query):
    prompt = f"""Bạn là chuyên gia phân tích ngôn ngữ pháp lý.
Nhiệm vụ: Trích xuất TỪ KHÓA LÕI từ câu hỏi để tra cứu cơ sở dữ liệu.

QUY TẮC TỐI THƯỢNG (Thực hiện theo thứ tự ưu tiên):
1. QUY TẮC ĐỊNH NGHĨA (ƯU TIÊN 1): Nếu câu hỏi có dạng "A là gì?", "Thế nào là A?", "A được hiểu như thế nào?", BẮT BUỘC chỉ lấy đúng cụm danh từ "A" làm từ khóa (Ví dụ: "Vạch kẻ đường là gì?" -> "vạch kẻ đường").
2. CHUẨN HÓA THUẬT NGỮ: Dịch từ lóng sang thuật ngữ luật pháp.
   - "uống rượu bia", "say xỉn" -> "nồng độ cồn"
   - "xe máy" -> "xe mô tô, xe gắn máy"
   - "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu"
   - "bằng lái" -> "giấy phép lái xe"
3. BỎ QUA TỪ NHIỄU: Xóa bỏ các từ để hỏi và từ nối (bao nhiêu tiền, bị phạt, là gì, như thế nào, tham gia giao thông). TUYỆT ĐỐI KHÔNG tự ý chế thêm chữ vào từ khóa gốc của người dùng.

CHỈ TRẢ VỀ JSON: {{"entities": ["từ khóa 1"]}}. KHÔNG GIẢI THÍCH, KHÔNG VIẾT THÊM TEXT.
Câu hỏi: {query}"""
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    generated_ids = llm_model.generate(**inputs, max_new_tokens=64, temperature=0.01)
    output_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        json_match = re.search(r'\{.*\}', output_text.replace('\n', ''))
        if json_match:
            data = json.loads(json_match.group(0))
            return [str(e).lower().strip() for e in data.get("entities", [])]
    except Exception: pass
    return []

def retrieve_top_k_chunks(query_entities):
    """Tìm Top-K Chunks và GẮN THÊM TÊN VĂN BẢN (DOC_NAME)"""
    if not query_entities: return []
    query_embeddings = embedder.encode(query_entities)
    chunk_scores = {}
    similarities = cosine_similarity(query_embeddings, entity_embeddings)
    
    for i, q_entity in enumerate(query_entities):
        top_k_indices = similarities[i].argsort()[-TOP_K_ENTITIES:][::-1]
        for idx in top_k_indices:
            matched_entity = graph_entities[idx]
            sim_score = similarities[i][idx]
            if sim_score < 0.45: continue # Ngưỡng nhạy hơn để bắt từ đồng nghĩa
            
            connected_chunks = tri_graph["entities"].get(matched_entity, [])
            for chunk_id in connected_chunks:
                if chunk_id not in chunk_scores: chunk_scores[chunk_id] = 0
                chunk_scores[chunk_id] += sim_score

    sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    top_chunks_ids = [c[0] for c in sorted_chunks[:TOP_K_CHUNKS]]
    
    # [CẬP NHẬT QUAN TRỌNG]: Lấy tên văn bản từ Document Node ghép vào Chunk Node
    results = []
    for cid in top_chunks_ids:
        chunk_data = tri_graph["chunks"][cid]
        # Tách ID để lấy tiền tố Document (Ví dụ: 35_2024_QH15)
        doc_id = cid.split("_D")[0] if "_D" in cid else cid.split("_Điều")[0]
        # Tra cứu tên luật từ Document Node
        doc_name = tri_graph["documents"].get(doc_id, {}).get("doc_name", "Văn bản pháp luật")
        
        results.append({
            "doc_name": doc_name,
            "dieu": chunk_data['hierarchy'].get('dieu', ''),
            "content": chunk_data['content']
        })
    return results

def generate_final_answer_stream(query, context_chunks):
    """Ép LLM trả lời và hiệu ứng gõ phím (Streaming)"""
    if not context_chunks:
        print("\n⚖️  Luật sư AI:\nRất tiếc, tôi không tìm thấy quy định pháp luật nào khớp với câu hỏi của bạn.")
        return
        
    context_text = "\n\n".join([f"- Nguồn: {c['doc_name']}\n  Quy định tại {c['dieu']}: {c['content']}" for c in context_chunks])
    
    prompt = f"""Bạn là một Luật sư AI tư vấn luật giao thông Việt Nam.
Hãy trả lời câu hỏi của người dùng DỰA HOÀN TOÀN VÀO Căn cứ pháp lý dưới đây.

QUY TẮC BẮT BUỘC:
1. PHÂN LOẠI ĐỐI TƯỢNG RÕ RÀNG: BẮT BUỘC chia các gạch đầu dòng và liệt kê riêng mức phạt cho TỪNG LOẠI XE.
2. TRÍCH DẪN NGUỒN: Phải ghi rõ [Điều, Khoản - Tên văn bản] ngay cạnh mỗi mức phạt.
3. KHÔNG BỊA ĐẶT: Nếu căn cứ bị thiếu một khung phạt nào đó, cứ lờ nó đi, tuyệt đối không được tự bịa ra số tiền.

[CĂN CỨ PHÁP LÝ TÌM ĐƯỢC]:
{context_text}

[CÂU HỎI CỦA NGƯỜI DÙNG]: {query}
[TRẢ LỜI]:"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    # Cấu hình Streamer để nhả từng chữ
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=1500, # Đã tăng giới hạn để không bị cắt cụt
        temperature=0.1,
        streamer=streamer
    )
    
    # Chạy LLM trong một luồng (thread) riêng để không khóa màn hình
    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()
    
    print("\n⚖️  Luật sư AI: ", end="")
    for new_text in streamer:
        print(new_text, end="", flush=True) # Nhả chữ trực tiếp ra màn hình
    print("\n")

# ==========================================
# 3. GIAO DIỆN CHAT 
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 LUẬT SƯ AI ĐÃ SẴN SÀNG CHAT!")
    print("="*50)
    
    while True:
        query = input("\n🧑 Bạn: ")
        if query.lower() in ['exit', 'quit']: break
        
        start_time = time.time()
        
        q_entities = extract_query_entities(query)
        print(f"   🔍 [Từ khóa bắt được] : {q_entities}")
        
        retrieved_chunks = retrieve_top_k_chunks(q_entities)
        print(f"   📚 [Đã truy xuất]     : Lấy ra {len(retrieved_chunks)} đoạn luật.")
        
        # Gọi hàm Streaming mới
        generate_final_answer_stream(query, retrieved_chunks)
        
        print(f"   ⏱️ [Tổng thời gian]   : {time.time() - start_time:.2f} giây")
        print("-" * 50)