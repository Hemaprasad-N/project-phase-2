import fitz  # PyMuPDF

class DocumentProcessor:
    @staticmethod
    def process_text(text: str) -> str:
        return text

    @staticmethod
    def process_pdf(file) -> str:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    @staticmethod
    def process_file(uploaded_file) -> str:
        if uploaded_file.name.endswith('.pdf'):
            return DocumentProcessor.process_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            return DocumentProcessor.process_text(uploaded_file.getvalue().decode('utf-8'))
        return ""
