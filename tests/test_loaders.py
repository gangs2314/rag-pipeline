"""Comprehensive tests for document loaders."""

import pytest
import json
import tempfile
from pathlib import Path

from rag_pipeline.loaders import (
    DocumentLoaderFactory,
    TextDocumentLoader,
    JSONDocumentLoader,
    CodeDocumentLoader,
)

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document


class TestTextDocumentLoader:
    """Test plain text document loader."""

    def test_load_simple_text(self):
        """Test loading a simple text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, world!\nThis is a test.")
            temp_path = f.name

        try:
            loader = TextDocumentLoader(Path(temp_path), "txt")
            docs = loader.load()

            assert len(docs) == 1
            assert "Hello, world!" in docs[0].page_content
            assert docs[0].metadata["source_file"] == Path(temp_path).name
            assert docs[0].metadata["doc_type"] == "txt"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_markdown(self):
        """Test loading markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Header\n\nContent here")
            temp_path = f.name

        try:
            loader = TextDocumentLoader(Path(temp_path), "md")
            docs = loader.load()

            assert len(docs) == 1
            assert "# Header" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            loader = TextDocumentLoader(Path(temp_path), "txt")
            docs = loader.load()

            assert len(docs) == 1
            assert docs[0].page_content == ""
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_unicode_content(self):
        """Test loading file with unicode characters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Hello 世界 🌍 Привет")
            temp_path = f.name

        try:
            loader = TextDocumentLoader(Path(temp_path), "txt")
            docs = loader.load()

            assert len(docs) == 1
            assert "世界" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestJSONDocumentLoader:
    """Test JSON document loader."""

    def test_load_simple_json(self):
        """Test loading simple JSON object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {"name": "John", "age": 30}
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = JSONDocumentLoader(Path(temp_path), "json")
            docs = loader.load()

            assert len(docs) == 1
            assert "name: John" in docs[0].page_content
            assert "age: 30" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_nested_json(self):
        """Test loading nested JSON structure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "user": {
                    "name": "Alice",
                    "details": {"age": 25, "city": "NYC"}
                }
            }
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = JSONDocumentLoader(Path(temp_path), "json")
            docs = loader.load()

            assert len(docs) == 1
            assert "Alice" in docs[0].page_content
            assert "NYC" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_json_array(self):
        """Test loading JSON array."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = [{"id": 1}, {"id": 2}]
            json.dump(data, f)
            temp_path = f.name

        try:
            loader = JSONDocumentLoader(Path(temp_path), "json")
            docs = loader.load()

            assert len(docs) == 1
            assert "id" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json}")
            temp_path = f.name

        try:
            loader = JSONDocumentLoader(Path(temp_path), "json")

            with pytest.raises(ValueError, match="Invalid JSON"):
                loader.load()
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestCodeDocumentLoader:
    """Test code document loader."""

    def test_load_python_code(self):
        """Test loading Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def hello():\n    return 'world'")
            temp_path = f.name

        try:
            loader = CodeDocumentLoader(Path(temp_path), "py")
            docs = loader.load()

            assert len(docs) == 1
            assert "def hello" in docs[0].page_content
            assert docs[0].metadata["language"] == "python"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_javascript_code(self):
        """Test loading JavaScript file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write("function hello() { return 'world'; }")
            temp_path = f.name

        try:
            loader = CodeDocumentLoader(Path(temp_path), "js")
            docs = loader.load()

            assert len(docs) == 1
            assert docs[0].metadata["language"] == "javascript"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_unsupported_code(self):
        """Test loading code with unknown extension."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("code")
            temp_path = f.name

        try:
            loader = CodeDocumentLoader(Path(temp_path), "xyz")
            docs = loader.load()

            assert len(docs) == 1
            assert docs[0].metadata["language"] == "unknown"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDocumentLoaderFactory:
    """Test document loader factory."""

    def test_factory_text_file(self):
        """Test factory creates correct loader for text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            docs = DocumentLoaderFactory.load_document(Path(temp_path))

            assert len(docs) == 1
            assert "test content" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_factory_markdown_file(self):
        """Test factory creates correct loader for markdown file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title")
            temp_path = f.name

        try:
            docs = DocumentLoaderFactory.load_document(Path(temp_path))

            assert len(docs) == 1
            assert "# Title" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_factory_json_file(self):
        """Test factory creates correct loader for JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            temp_path = f.name

        try:
            docs = DocumentLoaderFactory.load_document(Path(temp_path))

            assert len(docs) == 1
            assert "value" in docs[0].page_content
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_factory_unsupported_type(self):
        """Test factory raises error for unsupported file type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("test")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                DocumentLoaderFactory.load_document(Path(temp_path))
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_factory_nonexistent_file(self):
        """Test factory handles nonexistent file."""
        fake_path = Path("/nonexistent/file.txt")

        with pytest.raises(Exception):
            DocumentLoaderFactory.load_document(fake_path)

    def test_factory_get_loader(self):
        """Test factory get_loader method."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            loader = DocumentLoaderFactory.get_loader(Path(temp_path))

            assert isinstance(loader, TextDocumentLoader)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_factory_multiple_extensions(self):
        """Test factory handles multiple supported extensions."""
        extensions = [".txt", ".md", ".json", ".py", ".js"]

        for ext in extensions:
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                if ext == ".json":
                    json.dump({"test": "data"}, f)
                else:
                    f.write("test content")
                temp_path = f.name

            try:
                docs = DocumentLoaderFactory.load_document(Path(temp_path))
                assert len(docs) > 0
            finally:
                Path(temp_path).unlink(missing_ok=True)
