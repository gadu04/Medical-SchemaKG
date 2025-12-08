import pandas as pd
from openai import OpenAI
import json
import re
import os
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH
# ==============================================================================
# Kết nối LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Đường dẫn file
NODES_FILE = 'Eval/import/data/neo4j_nodes.csv'
RELS_FILE = 'Eval/import/data/neo4j_relationships.csv'
INPUT_QA_FILE = 'Eval/data/medquad.csv' 
OUTPUT_FILE = 'Eval/data/advanced_filtered_qa.csv'

# Cấu hình lọc
MAX_QUESTIONS = 700  # Số lượng câu hỏi tối đa muốn lấy
MIN_NODE_LENGTH = 4  # Chỉ lấy từ khóa dài > 3 ký tự để tránh nhiễu

# ==============================================================================
# 2. CHUẨN BỊ DỮ LIỆU TỪ KG (NODES & RELATIONSHIPS)
# ==============================================================================
print("⏳ [Bước 1] Đang xây dựng bộ từ điển từ Knowledge Graph...")

try:
    # --- Đọc file ---
    nodes_df = pd.read_csv(NODES_FILE)
    rels_df = pd.read_csv(RELS_FILE)

    # --- 1. Xác định các Node có quan hệ (Connected Nodes) ---
    # Lấy tập hợp tất cả ID xuất hiện ở cột START hoặc END trong file Relationships
    connected_ids = set(rels_df[':START_ID']).union(set(rels_df[':END_ID']))
    
    print(f"   - Tổng số Nodes gốc: {len(nodes_df)}")
    print(f"   - Số Nodes có quan hệ (được giữ lại): {len(connected_ids)}")

    # --- 2. Xử lý Nodes ---
    # Làm sạch tên node
    nodes_df['clean_name'] = nodes_df['name'].astype(str).str.replace(r'^\[Event:\s*|\]$', '', regex=True).str.lower().str.strip()
    
    # Tạo map ID -> Tên (dùng cho việc tạo Pair ở dưới)
    id_to_name = dict(zip(nodes_df[':ID'], nodes_df['clean_name']))
    
    # TẠO TẬP TỪ KHÓA (CHỈ LẤY NODE CÓ QUAN HỆ)
    kg_keywords = set()
    for _, row in nodes_df.iterrows():
        # LOGIC MỚI: Chỉ thêm vào từ điển nếu ID nằm trong danh sách connected_ids
        if row[':ID'] in connected_ids:
            name = row['clean_name']
            if len(name) >= MIN_NODE_LENGTH:
                kg_keywords.add(name)

    # --- 3. Xử lý Relationships ---
    kg_pairs = []
    for _, row in rels_df.iterrows():
        start_name = id_to_name.get(row[':START_ID'])
        end_name = id_to_name.get(row[':END_ID'])
        
        # Chỉ lấy cặp quan hệ nếu cả 2 đều có tên hợp lệ
        if start_name and end_name and len(start_name) >= MIN_NODE_LENGTH and len(end_name) >= MIN_NODE_LENGTH:
            kg_pairs.append((start_name, end_name))

    print(f"✅ Dữ liệu KG sau lọc: {len(kg_keywords)} từ khóa (chỉ nodes có qhệ), {len(kg_pairs)} cặp quan hệ.")

except Exception as e:
    print(f"❌ Lỗi đọc file KG: {e}")
    exit()

# ==============================================================================
# 3. BỘ LỌC THÔ (SCORING & RANKING)
# ==============================================================================
print(f"\n⏳ [Bước 2] Đang chấm điểm độ phù hợp của câu hỏi trong {INPUT_QA_FILE}...")

try:
    qa_df = pd.read_csv(INPUT_QA_FILE)
    # Tìm cột
    q_col = next((c for c in qa_df.columns if 'question' in c.lower()), qa_df.columns[0])
    a_col = next((c for c in qa_df.columns if 'answer' in c.lower()), qa_df.columns[1] if len(qa_df.columns)>1 else None)
except Exception as e:
    print(f"❌ Lỗi đọc file QA: {e}")
    exit()

def calculate_relevance_score(text):
    if not isinstance(text, str): return 0, ""
    text_lower = text.lower()
    score = 0
    reason = ""

    # Tiêu chí 1: Chứa CẶP QUAN HỆ (Strong Match) - 10 điểm
    # Ưu tiên cao nhất vì KG chắc chắn có thông tin liên kết
    for start, end in kg_pairs:
        if start in text_lower and end in text_lower:
            return 10, f"Strong Match: '{start}' & '{end}'" # Return luôn để tối ưu tốc độ

    # Tiêu chí 2: Chứa NODE (Weak Match) - 1 điểm
    # Duyệt qua keywords
    for k in kg_keywords:
        # Thêm space để tránh match 1 phần từ (vd: 'flu' trong 'influence')
        if f" {k} " in f" {text_lower} ":
            return 1, f"Keyword Match: '{k}'"
            
    return 0, ""

# Áp dụng chấm điểm (Dùng tqdm)
tqdm.pandas()
qa_df[['relevance_score', 'match_reason']] = qa_df[q_col].progress_apply(lambda x: pd.Series(calculate_relevance_score(x)))

# Lấy các ứng viên: Điểm cao trước, sau đó đến điểm thấp
candidates = qa_df[qa_df['relevance_score'] > 0].sort_values(by='relevance_score', ascending=False)

print(f"✅ [Bước 2 Xong] Tìm thấy {len(candidates)} câu hỏi tiềm năng.")
print(f"   - Strong matches (Score 10): {len(candidates[candidates['relevance_score'] == 10])}")
print(f"   - Weak matches (Score 1): {len(candidates[candidates['relevance_score'] == 1])}")

# ==============================================================================
# 4. BỘ LỌC TINH (SEMANTIC CHECK BẰNG LLAMA 3.1)
# ==============================================================================
# Chỉ lấy top N câu hỏi tốt nhất để check bằng AI (để tiết kiệm thời gian)
candidates_to_process = candidates.head(MAX_QUESTIONS).copy()

print(f"\n⏳ [Bước 3] Dùng Llama 3.1 kiểm tra ngữ nghĩa {len(candidates_to_process)} câu hỏi tốt nhất...")

final_results = []

def is_invalid_answer(answer):
    """Loại answer nếu trống hoặc bắt đầu bằng Key Points"""
    if not isinstance(answer, str):
        return True
    clean = answer.strip().lower()
    if clean == "":
        return True
    if clean.startswith("key points"):
        return True
    return False

def verify_relevance_with_llm(question, reason):
    # Prompt thông minh hơn: Yêu cầu AI đóng vai chuyên gia đánh giá
    prompt = f"""
    Task: Verify if the Question is medically relevant to the extracted Concept/Context from our Database.

    Context from Database: {reason}
    Question: "{question}"

    Analyze: Does the question meaningfully ask about the medical concepts identified in the Context?
    Return strictly JSON: {{"is_relevant": true}} or {{"is_relevant": false}}
    """
    
    try:
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a strict medical data validator. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if '"is_relevant": true' in content.lower():
            return True
        return False
    except:
        return False # Mặc định bỏ qua nếu lỗi

for index, row in tqdm(candidates_to_process.iterrows(), total=len(candidates_to_process)):
    question = row[q_col]
    answer = row[a_col] if a_col else ""
    reason = row['match_reason']

    # ❗ BỔ SUNG: loại nếu answer trống hoặc bắt đầu bằng Key Points
    if is_invalid_answer(answer):
        continue
    
    # Check bằng AI
    if verify_relevance_with_llm(question, reason):
        final_results.append({
            "Question": question,
            "Answer": answer,
            "Match_Type": "Strong" if row['relevance_score'] == 10 else "Weak",
            "Match_Detail": reason
        })



# ==============================================================================
# 5. LƯU KẾT QUẢ
# ==============================================================================
if final_results:
    # 1. Tạo DataFrame tạm từ kết quả AI check
    df_temp = pd.DataFrame(final_results)
    
    print(f"\n⏳ [Bước Bổ sung] Đang lọc các câu trả lời lỗi trình bày (format)...")
    
    # 2. Định nghĩa các mẫu lỗi (Abnormal patterns)
    # - Chứa ký tự Tab (\t)
    mask_tab = df_temp['Answer'].str.contains('\t', na=False)
    # - Xuống dòng (\n) theo sau là hơn 4 khoảng trắng (lỗi thò thụt dòng)
    mask_weird_spacing = df_temp['Answer'].str.contains(r'\n\s{4,}', regex=True, na=False)
    # - Dính câu (dấu chấm liền kề chữ Hoa): vd "end.The"
    mask_glued = df_temp['Answer'].str.contains(r'(?<=[a-z])\.[A-Z]', regex=True, na=False)

    # 3. Gom tất cả lỗi lại
    mask_to_remove = mask_tab | mask_weird_spacing | mask_glued
    
    # 4. Lọc bỏ và giữ lại dữ liệu sạch
    df_out = df_temp[~mask_to_remove].copy() # <--- QUAN TRỌNG: Lấy phần bù (~) của lỗi

    print(f"   - Tổng số câu sau AI check: {len(df_temp)}")
    print(f"   - Đã loại bỏ: {mask_to_remove.sum()} câu bị lỗi format.")
    print(f"   - Còn lại: {len(df_out)} câu hỏi sạch.")

    # 5. Lưu file (Chỉ lưu nếu còn dữ liệu)
    if not df_out.empty:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        # Chỉ lưu 2 cột chính
        df_save = df_out[['Question', 'Answer']]
        df_save.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        
        print(f"\n🎉 HOÀN TẤT! Đã lọc được {len(df_out)} câu hỏi chất lượng cao.")
        print(f"💾 File lưu tại: {OUTPUT_FILE}")
        
        # Thống kê
        print("🔍 Thống kê loại Match:")
        if 'Match_Type' in df_out.columns:
            print(df_out['Match_Type'].value_counts())
        
        print("\n🔍 5 Ví dụ đầu tiên:")
        # Chỉ hiện cột Match_Detail nếu nó tồn tại để debug
        cols_to_show = ['Question', 'Match_Detail'] if 'Match_Detail' in df_out.columns else ['Question']
        print(df_out[cols_to_show].head())
    else:
        print("\n⚠ Tất cả câu hỏi đã bị loại bỏ bởi bộ lọc format (Tab/Spacing/Glued words).")

else:
    print("\n⚠ Không có câu hỏi nào vượt qua bài kiểm tra ngữ nghĩa (AI Check).")