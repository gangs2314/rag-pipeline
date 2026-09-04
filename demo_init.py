"""Initialize demo with sample data on container startup."""

import json
import tempfile
from pathlib import Path
from rag_pipeline.pipeline import get_pipeline

def init_demo():
    """Load sample documents into the pipeline."""
    sample_docs_path = Path("/app/sample_data/documents.json")

    if not sample_docs_path.exists():
        print("Sample data not found, skipping demo init")
        return

    print("Initializing RAG Pipeline with sample data...")

    try:
        with open(sample_docs_path) as f:
            documents = json.load(f)

        pipeline = get_pipeline()

        # Create temporary files for each document
        temp_dir = Path(tempfile.gettempdir()) / "rag_demo"
        temp_dir.mkdir(exist_ok=True)

        ingested_count = 0
        for doc in documents[:5]:  # Load first 5 documents for demo
            doc_path = temp_dir / doc["filename"]
            doc_path.write_text(doc["content"])

            try:
                result = pipeline.ingest_document(doc_path)
                if result["status"] == "success":
                    ingested_count += 1
                    print(f"✓ Ingested {doc['filename']}: {result['chunk_count']} chunks")
                else:
                    print(f"✗ Failed to ingest {doc['filename']}")
            except Exception as e:
                print(f"✗ Error ingesting {doc['filename']}: {e}")

        stats = pipeline.get_stats()
        print(f"\nDemo initialized successfully!")
        print(f"Total documents: {stats['document_count']}")
        print(f"Ready to query at http://localhost:8000/docs")

    except Exception as e:
        print(f"Error initializing demo: {e}")

if __name__ == "__main__":
    init_demo()
