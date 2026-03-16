import json
import time
import re
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH QWEN (HUGGING FACE)
# ==========================================
print("⏳ Đang nạp mô hình Qwen vào GPU A40...")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# [QUAN TRỌNG] Cấu hình Tokenizer để chạy Batching
tokenizer.padding_side = "left" # Bắt buộc đệm bên trái cho generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto" # Tự động phân bổ VRAM
)
print("✅ Nạp mô hình thành công!\n" + "="*40)

# ==========================================
# 2. HÀM CHUẨN BỊ PROMPT CHO CẢ MẺ (BATCH)
# ==========================================
def prepare_batch_prompts(chunks):
    system_prompt = """Bạn là chuyên gia trích xuất dữ liệu pháp lý. Nhiệm vụ: Trích xuất TẤT CẢ Thực thể ngữ nghĩa từ văn bản.
QUY TẮC QUAN TRỌNG:
1. Đọc TOÀN BỘ câu, tuyệt đối KHÔNG BỎ SÓT các phần điều kiện (nếu, khi...) hoặc ngoại lệ (trừ trường hợp...).
2. Chỉ lấy: Đối tượng/Chủ thể (VD: xe mô tô, xe ưu tiên), Hành vi (VD: đi ngược chiều, làm nhiệm vụ), Khái niệm/Địa điểm (VD: đường cao tốc), Chế tài/Giấy tờ.
3. Không lấy từ nối, động từ chung chung. Giữ nguyên cụm từ dài nếu có nghĩa.
CHỈ TRẢ VỀ JSON hợp lệ có dạng: {"entities": ["thực thể 1", "thực thể 2"]}. KHÔNG GIẢI THÍCH."""

    formatted_prompts = []
    for chunk in chunks:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Đoạn văn bản:\n{chunk['content']}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)
        
    return formatted_prompts

# ==========================================
# 3. HÀM XÂY DỰNG TRI-GRAPH
# ==========================================
def build_full_tri_graph_batch(master_db_path, output_graph_path):
    print(f"📖 Đang đọc cơ sở dữ liệu từ: {master_db_path}")
    with open(master_db_path, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    # Khởi tạo đồ thị
    tri_graph = {
        "documents": {},
        "chunks": {},
        "entities": {}
    }

    # Lưu trước Document và Chunk Node
    for chunk in master_data:
        chunk_id = chunk["chunk_id"]
        tri_graph["chunks"][chunk_id] = {
            "hierarchy": chunk["hierarchy"],
            "content": chunk["content"]
        }
        doc_id = chunk_id.split("_D")[0] if "_D" in chunk_id else chunk_id.split("_Điều")[0]
        if doc_id not in tri_graph["documents"]:
            tri_graph["documents"][doc_id] = chunk.get("document_info", {})

    print("\n🚀 BẮT ĐẦU TRÍCH XUẤT THỰC THỂ (BATCH PROCESSING)...")
    
    # THÔNG SỐ TỐI ƯU CHO A40 (48GB VRAM)
    # Nếu máy báo Out of Memory (OOM), hãy giảm số này xuống 32 hoặc 16.
    # Nếu VRAM còn dư nhiều, có thể tăng lên 64 hoặc 128.
    BATCH_SIZE = 32 
    total_chunks = len(master_data)
    
    # Thanh tiến trình
    pbar = tqdm(total=total_chunks, desc="Xử lý Chunks", unit="chunk")

    # Vòng lặp cắt dữ liệu thành từng mẻ
    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = master_data[i : i + BATCH_SIZE]
        prompts = prepare_batch_prompts(batch_chunks)
        
        # Tiền xử lý đống text thành Tensor ném vào GPU
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        try:
            # Chạy suy luận song song cho cả Batch
            with torch.no_grad(): # Tắt tính đạo hàm để tiết kiệm RAM
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.01,
                    do_sample=True
                )
            
            # Cắt bỏ phần input (prompt) khỏi kết quả sinh ra
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Giải mã Tensor thành Text
            responses = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
            
            # Xử lý kết quả JSON của từng câu trong mẻ
            for j, response_text in enumerate(responses):
                chunk_id = batch_chunks[j]["chunk_id"]
                
                json_match = re.search(r'\{.*\}', response_text.replace('\n', ''))
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        if "entities" in data and isinstance(data["entities"], list):
                            entities = [str(e).lower().strip() for e in data["entities"] if len(str(e)) > 2]
                            
                            # Nhét vào Đồ thị
                            for entity in entities:
                                if entity not in tri_graph["entities"]:
                                    tri_graph["entities"][entity] = []
                                if chunk_id not in tri_graph["entities"][entity]:
                                    tri_graph["entities"][entity].append(chunk_id)
                    except json.JSONDecodeError:
                        pass # Bỏ qua nếu LLM sinh JSON lỗi

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️ LỖI TRÀN VRAM (OOM) ở mẻ {i}. Hãy giảm BATCH_SIZE xuống!")
                torch.cuda.empty_cache() # Dọn dẹp VRAM để không sập script
            else:
                print(f"\n⚠️ Lỗi không xác định: {e}")

        # Cập nhật thanh tiến trình
        pbar.update(len(batch_chunks))

        # AUTO-SAVE sau mỗi vài mẻ (Khoảng ~150 chunks một lần lưu)
        if (i // BATCH_SIZE) % 5 == 0:
            with open(output_graph_path, 'w', encoding='utf-8') as f:
                json.dump(tri_graph, f, ensure_ascii=False, indent=4)

    pbar.close()

    # Lưu file Final
    with open(output_graph_path, 'w', encoding='utf-8') as f:
        json.dump(tri_graph, f, ensure_ascii=False, indent=4)
        
    print(f"\n🎉 HOÀN TẤT! Đồ thị Tri-Graph hoàn chỉnh đã lưu tại: {output_graph_path}")
    print(f"📊 Báo cáo: {len(tri_graph['documents'])} Documents | {len(tri_graph['chunks'])} Chunks | {len(tri_graph['entities'])} Entities")

# ==========================================
# KHỞI CHẠY
# ==========================================
if __name__ == "__main__":
    PROJECT_ROOT = "/workspace/RAG-System" 
    MASTER_DB = os.path.join(PROJECT_ROOT, "metadata/master_rag_database.json")
    GRAPH_OUTPUT = os.path.join(PROJECT_ROOT, "metadata/linear_tri_graph.json")
    
    os.makedirs(os.path.dirname(GRAPH_OUTPUT), exist_ok=True)
    build_full_tri_graph_batch(MASTER_DB, GRAPH_OUTPUT)