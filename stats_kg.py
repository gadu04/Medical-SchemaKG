import pandas as pd
import os

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
NODES_FILE = 'Eval/import/data/neo4j_nodes.csv'
RELS_FILE = 'Eval/import/data/neo4j_relationships.csv'

# Con số bạn muốn hiển thị trên Slide (nếu muốn để máy tự đếm thì đặt là None)
FORCED_CHUNK_COUNT = 405 

def generate_stats():
    print("📊 ĐANG TÍNH TOÁN THỐNG KÊ KNOWLEDGE GRAPH (FINAL)...\n")
    
    total_nodes = 0
    entity_count = 0
    event_count = 0
    rels_count = 0
    unique_chunks_from_data = 0

    # 1. Xử lý Nodes
    if os.path.exists(NODES_FILE):
        try:
            nodes_df = pd.read_csv(NODES_FILE)
            total_nodes = len(nodes_df)
            
            # Sửa lỗi đọc cột Label: Tìm cột 'labels' (viết thường)
            if 'labels' in nodes_df.columns:
                type_counts = nodes_df['labels'].value_counts()
                entity_count = type_counts.get('Entity', 0)
                event_count = type_counts.get('Event', 0)
            else:
                print(f"⚠ Không tìm thấy cột 'labels'.")
        except Exception as e:
            print(f"❌ Lỗi đọc file Nodes: {e}")
    else:
        print(f"❌ Không tìm thấy file: {NODES_FILE}")

    # 2. Xử lý Relationships & Chunks
    if os.path.exists(RELS_FILE):
        try:
            rels_df = pd.read_csv(RELS_FILE)
            rels_count = len(rels_df)
            
            # Đếm số Segment/Chunk thực tế tham gia vào quan hệ
            if 'segment_id' in rels_df.columns:
                unique_chunks_from_data = rels_df['segment_id'].nunique()
        except Exception as e:
            print(f"❌ Lỗi đọc file Relationships: {e}")
    else:
        print(f"❌ Không tìm thấy file: {RELS_FILE}")

    # Quyết định số lượng Chunk để hiển thị
    display_chunks = FORCED_CHUNK_COUNT if FORCED_CHUNK_COUNT else unique_chunks_from_data

    # 3. Xuất bảng kết quả ĐẸP để chụp Slide
    print("\n" + "="*60)
    print(f"{'BẢNG THỐNG KÊ DỮ LIỆU MEDICAL-SCHEMAKG':^60}")
    print("="*60)
    print(f"{'Thành phần (Metric)':<35} | {'Số lượng (Count)':<20}")
    print("-" * 60)
    
    # Phần 1: Dữ liệu nguồn
    print(f"{'Text Chunks (Đoạn văn bản)':<35} | {display_chunks:,}")
    print("-" * 60)
    
    # Phần 2: Dữ liệu Graph
    print(f"{'Tổng số Nodes (Total Nodes)':<35} | {total_nodes:,}")
    print(f"{'  ├── Entities (Thực thể)':<35} | {entity_count:,}")
    print(f"{'  └── Events (Sự kiện)':<35} | {event_count:,}")
    print("-" * 60)
    print(f"{'Tổng số Relationships (Cạnh)':<35} | {rels_count:,}")
    print("="*60)

    # 4. Gợi ý biểu đồ
    if total_nodes > 0:
        ent_pct = (entity_count / total_nodes) * 100
        evt_pct = (event_count / total_nodes) * 100
        print(f"\n SỐ LIỆU VẼ BIỂU ĐỒ (PIE CHART):")
        print(f"   - Entity: {ent_pct:.1f}%")
        print(f"   - Event:  {evt_pct:.1f}%")
        
    # In thêm thông tin debug nhỏ bên dưới
    if FORCED_CHUNK_COUNT and unique_chunks_from_data != FORCED_CHUNK_COUNT:
        print(f"\n(Note: Thực tế file quan hệ chứa {unique_chunks_from_data} chunk unique, nhưng bảng hiển thị {FORCED_CHUNK_COUNT} theo cấu hình)")

if __name__ == "__main__":
    generate_stats()