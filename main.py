"""
Medical-SchemaKG Framework - Main Orchestrator
================================================
Entry point for the four-phase pipeline that builds a Knowledge Graph from medical text.

Pipeline Phases:
1. Document Ingestion & Preprocessing (Stubbed)
2. Triple Extraction (Core Module)
3. Hybrid Schema Induction & Ontology Grounding (Partial Stub)
4. Knowledge Graph Construction (Core Module)
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment variables from {env_file}")
    else:
        print(f"ℹ No .env file found. Using system environment variables.")
except ImportError:
    print("ℹ python-dotenv not installed. Using system environment variables only.")
    print("  Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠ Warning: Could not load .env file: {e}")

from pipeline.phase_1_ingestion import load_and_segment_text
from pipeline.phase_2_triple_extraction import TripleExtractor
from pipeline.phase_3_schema_induction import dynamically_induce_concepts, ground_concepts_to_ontology
from pipeline.phase_4_kg_construction import build_knowledge_graph, export_graph_to_neo4j_csv
from utils.visualization import print_pipeline_summary, save_graph_visualization


def main():
    """
    Main orchestrator function that executes the four-phase pipeline.
    """
    print("=" * 80)
    print("MEDICAL-SCHEMAKG FRAMEWORK - PIPELINE EXECUTION")
    print("=" * 80)
    print()
    
    # Load configuration from environment variables (set in .env file)
    use_real_llm = os.getenv("USE_REAL_LLM", "false").lower() == "true"
    input_file = os.getenv("INPUT_FILE", "data/parsed/AMA_Family_Guide_content.md")
    output_dir = os.getenv("OUTPUT_DIR", "output")
    model_name = os.getenv("MODEL_NAME", "local-model")
    lm_studio_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    
    print(f"Configuration:")
    print(f"  - LLM Mode: {'REAL API (LM Studio)' if use_real_llm else 'STUBBED'}")
    if use_real_llm:
        print(f"  - LM Studio URL: {lm_studio_url}")
        print(f"  - Model: {model_name}")
    print(f"  - Input File: {input_file}")
    print(f"  - Output Directory: {output_dir}")
    print()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # PHASE 1: DOCUMENT INGESTION & PREPROCESSING (STUBBED)
    # =========================================================================
    print("-" * 80)
    print("PHASE 1: DOCUMENT INGESTION & PREPROCESSING")
    print("-" * 80)
    print("Status: Loading and segmenting medical text...")
    
    try:
        text_segments = load_and_segment_text(input_file)
        print(f"✓ Phase 1 Complete. Found {len(text_segments)} text chunks.")
        print(f"  Sample chunk: {text_segments[0]['text'][:100]}..." if text_segments else "  No chunks found.")
        print()
    except Exception as e:
        print(f"✗ Phase 1 Failed: {e}")
        return
    
    # =========================================================================
    # PHASE 2: TRIPLE EXTRACTION
    # =========================================================================
    print("-" * 80)
    print("PHASE 2: TRIPLE EXTRACTION")
    print("-" * 80)
    print("Status: Extracting (Head, Relation, Tail) triples from text...")
    
    try:
        extractor = TripleExtractor(use_real_llm=use_real_llm)
        all_triples, unique_nodes = extractor.extract_from_segments(text_segments)
        
        print(f"✓ Phase 2 Complete.")
        print(f"  - Total Triples Extracted: {len(all_triples)}")
        print(f"  - Entity-Entity (E-E): {sum(1 for t in all_triples if t['type'] == 'E-E')}")
        print(f"  - Entity-Event (E-Ev): {sum(1 for t in all_triples if t['type'] == 'E-Ev')}")
        print(f"  - Event-Event (Ev-Ev): {sum(1 for t in all_triples if t['type'] == 'Ev-Ev')}")
        print(f"  - Unique Nodes: {len(unique_nodes)}")
        print()
    except Exception as e:
        print(f"✗ Phase 2 Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # PHASE 3: HYBRID SCHEMA INDUCTION & ONTOLOGY GROUNDING
    # =========================================================================
    print("-" * 80)
    print("PHASE 3: HYBRID SCHEMA INDUCTION & ONTOLOGY GROUNDING")
    print("-" * 80)
    
    # Part 3a: Dynamic Induction (Core Module)
    print("Status: Part 3a - Dynamically inducing abstract concepts...")
    try:
        induced_concepts = dynamically_induce_concepts(
            unique_nodes, 
            all_triples=all_triples,
            use_real_llm=use_real_llm
        )
        print(f"✓ Part 3a Complete. Induced concepts for {len(induced_concepts)} nodes.")
        print()
    except Exception as e:
        print(f"✗ Part 3a Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Part 3b: Ontology Grounding (Stubbed)
    print("Status: Part 3b - Grounding concepts to medical ontologies...")
    try:
        grounded_nodes = ground_concepts_to_ontology(induced_concepts)
        print(f"✓ Part 3b Complete. Grounded {len(grounded_nodes)} nodes to ontology IDs.")
        print()
    except Exception as e:
        print(f"✗ Part 3b Failed: {e}")
        return
    
    # =========================================================================
    # PHASE 4: KNOWLEDGE GRAPH CONSTRUCTION
    # =========================================================================
    print("-" * 80)
    print("PHASE 4: KNOWLEDGE GRAPH CONSTRUCTION")
    print("-" * 80)
    print("Status: Building the final Onto-MedKG graph...")
    
    try:
        knowledge_graph = build_knowledge_graph(all_triples, grounded_nodes)
        print(f"✓ Phase 4 Complete.")
        print(f"  - Total Nodes: {knowledge_graph.number_of_nodes()}")
        print(f"  - Total Edges: {knowledge_graph.number_of_edges()}")
        print()
    except Exception as e:
        print(f"✗ Phase 4 Failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # FINALIZATION
    # =========================================================================
    print("=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print()
    
    # Print summary
    print_pipeline_summary(text_segments, all_triples, grounded_nodes, knowledge_graph)
    
    # Save visualization
    try:
        viz_path = os.path.join(output_dir, "knowledge_graph.png")
        save_graph_visualization(knowledge_graph, viz_path)
        print(f"\n✓ Knowledge graph visualization saved to: {viz_path}")
    except Exception as e:
        print(f"\n⚠ Could not save visualization: {e}")

    # Export Neo4j CSVs for bulk import
    try:
        neo4j_files = export_graph_to_neo4j_csv(knowledge_graph, output_dir)
        print(f"\n✓ Neo4j CSVs written:")
        print(f"  - Nodes: {neo4j_files.get('neo4j_nodes')}")
        print(f"  - Relationships: {neo4j_files.get('neo4j_rels')}")
    except Exception as e:
        print(f"\n⚠ Could not export Neo4j CSVs: {e}")
    
    print("\n" + "=" * 80)
    print("Thank you for using Medical-SchemaKG Framework!")
    print("=" * 80)


if __name__ == "__main__":
    main()
