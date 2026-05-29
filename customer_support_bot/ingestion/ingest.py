import argparse
from pathlib import Path

from customer_support_bot.ingestion.embedder import run_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for customer support bot")
    parser.add_argument("--filepath", required=True, help="Path to the document to ingest")
    parser.add_argument("--tenant-id", default="acme", help="Tenant ID to build index for")
    args = parser.parse_args()

    filepath = Path(args.filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")

    result = run_ingestion_pipeline(str(filepath), args.tenant_id)
    print(result)


if __name__ == "__main__":
    main()
