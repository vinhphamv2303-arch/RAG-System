import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def build_and_save_index(graph_path, vector_output_path):
    print("\n" + "="*50)
    print("🛠️ BƯỚC 1: NHÚNG VECTOR CHO 20.000+ THỰC THỂ")
    print("="*50)
    
    print("⏳ Đang tải Mô hình tiếng Việt (vietnamese-sbert)...")
    embedder = SentenceTransformer("keepitreal/vietnamese-sbert")

    with open(graph_path, 'r', encoding='utf-8') as f:
        tri_graph = json.load(f)

    graph_entities = list(tri_graph["entities"].keys())
    
    print(f"🧠 Đang nhúng {len(graph_entities)} Thực thể thành Vector...")
    entity_embeddings = embedder.encode(graph_entities, show_progress_bar=True)

    print(f"💾 Đang lưu Vector Database ra:\n   {vector_output_path}")
    np.save(vector_output_path, entity_embeddings)
    
    keys_path = vector_output_path.replace(".npy", "_keys.json")
    with open(keys_path, 'w', encoding='utf-8') as f:
        json.dump(graph_entities, f, ensure_ascii=False, indent=4)

    print("\n✅ HOÀN TẤT BƯỚC INDEX!")

if __name__ == "__main__":
    PROJECT_ROOT = "/workspace/RAG-System"
    GRAPH_PATH = os.path.join(PROJECT_ROOT, "metadata/linear_tri_graph.json")
    VECTOR_OUTPUT = os.path.join(PROJECT_ROOT, "metadata/entity_vectors.npy")
    
    os.makedirs(os.path.dirname(VECTOR_OUTPUT), exist_ok=True)
    build_and_save_index(GRAPH_PATH, VECTOR_OUTPUT)