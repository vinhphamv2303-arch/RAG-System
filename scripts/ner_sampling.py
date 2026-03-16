import json
import random
import re


def filter_and_split_ner_data(input_json_path, num_train=1200, num_test=300):
    print(f"📥 Đang nạp dữ liệu từ: {input_json_path}")

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file JSON đầu vào.")
        return

    valid_chunks = []

    # Các từ khóa thường xuất hiện ở các điều khoản thủ tục, không có giá trị trích xuất NER luật
    stop_phrases = [
        "hiệu lực thi hành", "tổ chức thực hiện", "chánh văn phòng",
        "trách nhiệm thi hành", "phạm vi điều chỉnh", "giải thích từ ngữ",
        "bãi bỏ", "ban hành kèm theo"
    ]

    for item in data:
        content = item.get("content", "").strip()
        chunk_id = item.get("chunk_id", "")

        content_lower = content.lower()

        # 1. BỘ LỌC ĐỘ DÀI (QUAN TRỌNG NHẤT)
        if len(content) < 60 or len(content) > 1200:
            continue

        # 2. BỘ LỌC TỪ KHÓA NHIỄU
        if any(phrase in content_lower for phrase in stop_phrases):
            continue

        # 3. BỘ LỌC ĐỊNH DẠNG (Bỏ các chunk chỉ chứa danh sách tên viết tắt, ký hiệu...)
        if not re.search(r'[a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', content_lower):
            continue

        # CẬP NHẬT: Vỏ rỗng với cấu trúc 11 nhãn hardcore
        valid_chunks.append({
            "chunk_id": chunk_id,
            "text": content,
            "entities": {
                "DOI_TUONG": [],
                "PHUONG_TIEN": [],
                "GIAY_TO": [],
                "HANH_VI": [],
                "HA_TANG": [],
                "THONG_SO": [],
                "TINH_TRANG": [],
                "MUC_PHAT_TIEN": [],
                "HINH_PHAT_BO_SUNG": [],
                "THOI_GIAN_PHAT": [],
                "BIEN_PHAP_KHAC_PHUC": []
            }
        })

    total_valid = len(valid_chunks)
    print(f"✅ Lọc thành công {total_valid} chunks đạt 'Tiêu chuẩn Vàng'.")

    total_needed = num_train + num_test
    if total_valid < total_needed:
        print(f"⚠️ CẢNH BÁO: Dữ liệu sạch ({total_valid}) không đủ số lượng yêu cầu ({total_needed}).")
        print("Sẽ sử dụng toàn bộ dữ liệu hiện có và chia theo tỷ lệ tương đối.")
        # Chia theo tỷ lệ 80/20 nếu thiếu data
        num_test = int(total_valid * 0.2)
        num_train = total_valid - num_test

    # Trộn ngẫu nhiên dữ liệu để đảm bảo tính phân phối đều
    random.seed(42)  # Cố định seed để nếu chạy lại code vẫn ra kết quả giống nhau
    random.shuffle(valid_chunks)

    # Cắt mảng
    train_data = valid_chunks[:num_train]
    test_data = valid_chunks[num_train:num_train + num_test]

    # Lưu file
    train_file = "../metadata/ner_train_1200.json"
    test_file = "../metadata/ner_test_300.json"

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print("-" * 30)
    print(f"🎯 ĐÃ CHIA XONG DỮ LIỆU TẬP TRUNG:")
    print(f"📚 Tập Train : {len(train_data)} mẫu -> Đã lưu vào {train_file}")
    print(f"🧪 Tập Test  : {len(test_data)} mẫu -> Đã lưu vào {test_file}")


# --- KHỞI CHẠY ---
if __name__ == "__main__":
    # Thay file này bằng file JSON master của bạn
    INPUT_FILE = "../metadata/master_rag_database.json"
    filter_and_split_ner_data(INPUT_FILE, num_train=1200, num_test=300)