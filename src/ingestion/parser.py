import os
from bs4 import BeautifulSoup
from pypdf import PdfReader

class AdvancedParser:
    def __init__(self):
        pass

    def parse_file(self, file_path: str) -> str:
        """
        Parses PDF, HTML, or raw text files and extracts the text content.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext in ['.html', '.htm']:
            return self._parse_html(file_path)
        else:
            return self._parse_txt(file_path)

    def _parse_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    def _parse_html(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator=' ', strip=True)

    def _parse_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
