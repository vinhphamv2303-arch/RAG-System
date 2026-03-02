import os
import glob
import json
import re
from datetime import datetime


def build_global_dictionary(preprocessed_dir):
    print(f"📚 Đang nạp dữ liệu từ {preprocessed_dir} vào bộ nhớ...")
    all_chunks = []
    # Vẫn giữ nguyên cấu trúc đọc file
    for filepath in glob.glob(os.path.join(preprocessed_dir, "*.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            doc_info = data.get("document_info", {})
            for chunk in data.get("chunks", []):
                chunk["document_info"] = doc_info
                all_chunks.append(chunk)

    print(f"✅ Đã nạp xong {len(all_chunks)} khối dữ liệu.")
    return all_chunks


def get_full_reference_text(target_id, all_chunks):
    """
    HÀM MỚI (PHÉP MÀU NẰM Ở ĐÂY):
    Lấy nội dung của ID mục tiêu VÀ toàn bộ con cháu (Khoản, Điểm) của nó.
    """
    collected_lines = []
    for chunk in all_chunks:
        cid = chunk["chunk_id"]
        # So khớp: Lấy chính nó HOẶC các cấp con (Dấu _ đảm bảo K1 không bắt nhầm K10)
        if cid == target_id or cid.startswith(target_id + "_"):
            collected_lines.append(chunk["content"])

    if not collected_lines:
        return None

    # Nối tất cả lại thành 1 đoạn văn bản liên tục
    return "\n".join(collected_lines)


def resolve_inline_references(all_chunks, log_filepath):
    print(f"🔄 Bắt đầu dò tìm tham chiếu và ghi log ra {log_filepath}...")
    resolved_count = 0

    with open(log_filepath, 'w', encoding='utf-8') as log_file:
        log_file.write(f"=== BÁO CÁO GIẢI THAM CHIẾU CHÉO ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")

        re_khoan_dieunay = r"(?i)khoản\s+(\d+)[,\s]+Điều\s+này"
        re_dieu_cung_vanban = r"(?i)(?:tại|theo quy định tại|áp dụng)\s+Điều\s+(\d+[a-zA-Z]*)"

        for chunk in all_chunks:
            chunk_id = chunk["chunk_id"]
            content = chunk["content"]
            id_parts = chunk_id.split("_")

            if len(id_parts) >= 4:
                doc_prefix = f"{id_parts[0]}_{id_parts[1]}_{id_parts[2]}"
                current_dieu_id = id_parts[3]
            else:
                continue

            # --- XỬ LÝ MẪU 1: "khoản X Điều này" ---
            matches_k = re.finditer(re_khoan_dieunay, content)
            for match in matches_k:
                khoan_target = match.group(1)
                target_id = f"{doc_prefix}_{current_dieu_id}_K{khoan_target}"

                # SỬ DỤNG HÀM GOM NHÁNH THAY VÌ TRA CỨU 1:1
                ref_text = get_full_reference_text(target_id, all_chunks)

                if ref_text:
                    # Dọn dẹp số đếm ở đầu câu đầu tiên cho đẹp
                    clean_ref_text = re.sub(r"^\d+\.\s*", "", ref_text, count=1)

                    replacement = f"{match.group(0)} [Nội dung: {clean_ref_text}]"
                    content = content.replace(match.group(0), replacement)
                    resolved_count += 1

                    log_file.write(f"📍 CHUNK GỐC  : {chunk_id}\n")
                    log_file.write(f"🔍 TÌM THẤY   : '{match.group(0)}'\n")
                    log_file.write(f"🔗 TRỎ TỚI ID : {target_id} (Và các điểm con)\n")
                    log_file.write(f"📝 ĐÃ CHÈN    : {clean_ref_text[:200]}... (Rút gọn log)\n")
                    log_file.write("-" * 60 + "\n")

            # --- XỬ LÝ MẪU 2: "tại Điều Y" ---
            matches_d = re.finditer(re_dieu_cung_vanban, content)
            for match in matches_d:
                dieu_target = match.group(1)
                target_dieu_id = f"D{dieu_target}"

                if target_dieu_id != current_dieu_id:
                    target_id = f"{doc_prefix}_{target_dieu_id}"

                    # SỬ DỤNG HÀM GOM NHÁNH
                    ref_text = get_full_reference_text(target_id, all_chunks)

                    if ref_text:
                        clean_ref_text = re.sub(r"^Điều\s+\d+[a-zA-Z]*\.\s*(.*?)\n", r"\1 - ", ref_text, count=1)
                        replacement = f"{match.group(0)} [Nội dung Điều {dieu_target}: {clean_ref_text}]"
                        content = content.replace(match.group(0), replacement)
                        resolved_count += 1

                        log_file.write(f"📍 CHUNK GỐC  : {chunk_id}\n")
                        log_file.write(f"🔍 TÌM THẤY   : '{match.group(0)}'\n")
                        log_file.write(f"🔗 TRỎ TỚI ID : {target_id} (Toàn bộ Điều)\n")
                        log_file.write(f"📝 ĐÃ CHÈN    : {clean_ref_text[:200]}... (Rút gọn log)\n")
                        log_file.write("-" * 60 + "\n")

            chunk["content"] = content

    print(f"🎯 Đã giải quyết và ghi log {resolved_count} điểm tham chiếu!")
    return all_chunks


def run_global_linker(preprocessed_dir, final_output_file, log_dir):
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, "cross_references_log.txt")

    all_chunks = build_global_dictionary(preprocessed_dir)
    if not all_chunks: return

    # Gọi hàm xử lý mới
    resolved_chunks = resolve_inline_references(all_chunks, log_file_path)

    output_dir = os.path.dirname(final_output_file)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(resolved_chunks, f, ensure_ascii=False, indent=4)

    print(f"🚀 XUẤT XƯỞNG! Cơ sở dữ liệu gộp tại: {final_output_file}")


# KHỞI CHẠY
if __name__ == "__main__":
    PROJECT_ROOT = "/home/vinh/projects/rag-system"
    PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/preprocessed")
    FINAL_DB_FILE = os.path.join(PROJECT_ROOT, "metadata/master_rag_database.json")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

    run_global_linker(PREPROCESSED_DIR, FINAL_DB_FILE, LOG_DIR)