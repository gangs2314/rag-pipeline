"""CLI commands for RAG pipeline."""

import sys
from pathlib import Path
from typing import Optional

import click

from rag_pipeline.pipeline import get_pipeline
from rag_pipeline.kaggle_integration import get_kaggle
from rag_pipeline.config import settings


@click.group()
def cli():
    """RAG Pipeline CLI - Document ingestion and semantic search."""
    pass


@cli.command()
@click.option(
    "--host",
    default="0.0.0.0",
    help="Server host",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Server port",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload on code changes",
)
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI server."""
    click.echo(f"Starting RAG Pipeline API on {host}:{port}")

    try:
        import uvicorn

        uvicorn.run(
            "rag_pipeline.api:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except Exception as e:
        click.secho(f"Error starting server: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
def ingest(file_path: str):
    """Ingest a document."""
    file_path = Path(file_path)

    if not file_path.exists():
        click.secho(f"File not found: {file_path}", fg="red")
        sys.exit(1)

    try:
        click.echo(f"Ingesting document: {file_path.name}")
        pipeline = get_pipeline()
        result = pipeline.ingest_document(file_path)

        if result["status"] == "error":
            click.secho(f"Error: {result.get('error')}", fg="red")
            sys.exit(1)

        click.secho("✓ Document ingested successfully", fg="green")
        click.echo(f"  Document ID: {result['document_id']}")
        click.echo(f"  Chunks created: {result['chunk_count']}")
        click.echo(f"  Chunks upserted: {result['upserted_count']}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--top-k",
    default=5,
    type=int,
    help="Number of top results to return",
)
def query(query: str, top_k: int):
    """Query documents."""
    try:
        click.echo(f"Query: {query}")
        pipeline = get_pipeline()
        result = pipeline.query(query, top_k)

        if result["status"] == "error":
            click.secho(f"Error: {result.get('error')}", fg="red")
            sys.exit(1)

        click.echo(f"\nFound {result['result_count']} results:\n")

        for idx, hit in enumerate(result["results"], 1):
            similarity = hit["similarity_score"]
            click.echo(f"{idx}. [Similarity: {similarity:.3f}]")
            click.echo(f"   Source: {hit['metadata'].get('source_file', 'N/A')}")
            if "page_number" in hit["metadata"]:
                click.echo(f"   Page: {hit['metadata']['page_number']}")
            click.echo(f"   Content: {hit['content'][:200]}...")
            click.echo()

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@cli.command()
def stats():
    """Show pipeline statistics."""
    try:
        pipeline = get_pipeline()
        stats_data = pipeline.get_stats()

        click.echo("\n=== RAG Pipeline Statistics ===\n")
        click.echo(f"Vector Store Type: {stats_data.get('vector_store_type')}")
        click.echo(f"Total Documents: {stats_data.get('document_count')}")
        click.echo(f"Registered Documents: {stats_data.get('registered_documents')}")
        click.echo(f"Embedding Model: {stats_data.get('embedding_model')}")

        chunking = stats_data.get("chunking_config", {})
        click.echo(f"\nChunking Configuration:")
        click.echo(f"  Chunk Size: {chunking.get('chunk_size')} tokens")
        click.echo(f"  Chunk Overlap: {chunking.get('chunk_overlap')} tokens")
        click.echo(f"  Parent-Child Chunking: {chunking.get('use_parent_child')}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@cli.group()
def kaggle():
    """Kaggle dataset commands."""
    pass


@kaggle.command(name="download")
@click.argument("dataset_name")
@click.option(
    "--output",
    type=click.Path(),
    help="Output directory",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force re-download",
)
def download_dataset(dataset_name: str, output: Optional[str], force: bool):
    """Download a dataset from Kaggle."""
    try:
        click.echo(f"Downloading dataset: {dataset_name}")
        kaggle = get_kaggle()
        result = kaggle.download_dataset(
            dataset_name,
            Path(output) if output else None,
            force=force,
        )

        if result["status"] == "error":
            click.secho(f"Error: {result.get('error')}", fg="red")
            sys.exit(1)

        click.secho("✓ Dataset downloaded successfully", fg="green")
        click.echo(f"  Path: {result.get('path')}")
        click.echo(f"  New download: {result.get('new_download')}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@kaggle.command(name="list")
def list_datasets():
    """List locally downloaded datasets."""
    try:
        kaggle = get_kaggle()
        datasets = kaggle.list_local_datasets()

        if not datasets:
            click.echo("No datasets found locally")
            return

        click.echo(f"\nFound {len(datasets)} local dataset(s):\n")
        for dataset in datasets:
            click.echo(f"• {dataset['name']}")
            click.echo(f"  Path: {dataset['path']}")
            click.echo(f"  Files: {dataset['file_count']}\n")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@kaggle.command(name="delete")
@click.argument("dataset_name")
def delete_dataset(dataset_name: str):
    """Delete a downloaded dataset."""
    if not click.confirm(f"Delete dataset: {dataset_name}?"):
        click.echo("Cancelled")
        return

    try:
        kaggle = get_kaggle()
        result = kaggle.delete_dataset(dataset_name)

        if result["status"] == "error":
            click.secho(f"Error: {result.get('error')}", fg="red")
            sys.exit(1)

        click.secho("✓ Dataset deleted", fg="green")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@cli.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "--ext",
    multiple=True,
    default=[".pdf", ".txt", ".md", ".json", ".docx"],
    help="File extensions to ingest",
)
def batch_ingest(directory: str, ext: tuple):
    """Batch ingest all documents in a directory."""
    dir_path = Path(directory)

    if not dir_path.is_dir():
        click.secho(f"Directory not found: {directory}", fg="red")
        sys.exit(1)

    try:
        click.echo(f"Scanning directory: {directory}")
        pipeline = get_pipeline()

        # Find all matching files
        files_to_ingest = []
        for extension in ext:
            files_to_ingest.extend(dir_path.glob(f"**/*{extension}"))

        if not files_to_ingest:
            click.echo("No matching files found")
            return

        click.echo(f"Found {len(files_to_ingest)} files to ingest\n")

        successful = 0
        failed = 0

        with click.progressbar(
            files_to_ingest,
            label="Ingesting",
            show_pos=True,
        ) as bar:
            for file_path in bar:
                result = pipeline.ingest_document(file_path)
                if result["status"] == "success":
                    successful += 1
                else:
                    failed += 1
                    click.echo(f"\nFailed: {file_path.name} - {result.get('error')}")

        click.secho(f"\n✓ Batch ingest complete", fg="green")
        click.echo(f"  Successful: {successful}")
        click.echo(f"  Failed: {failed}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    cli()
