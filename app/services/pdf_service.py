from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFService:

    def extract_pages(self, file_path: str):
        """
        Extract text from PDF page by page.
        """

        reader = PdfReader(file_path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "page_number": page_number,
                    "text": text.strip(),
                })

        return pages

    def create_chunks(
        self,
        pages: list[dict],
        document_id: int,
        filename: str,
    ):
        """
        Split each page into smaller chunks while
        preserving page metadata.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = []

        for page in pages:

            page_chunks = splitter.split_text(page["text"])

            for chunk_index, chunk_text in enumerate(page_chunks):

                chunks.append({
                    "chunk_id": f"{document_id}_{page['page_number']}_{chunk_index}",
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                })

        return chunks