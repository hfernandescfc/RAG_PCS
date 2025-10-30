from typing import Tuple, List, Optional

from langchain.docstore.document import Document
from .utils_io import verificar_poppler


def verificar_pdf_tem_texto(pdf_path: str) -> bool:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        for page in reader.pages[:3]:
            text = (page.extract_text() or "").strip()
            if len(text) > 100:
                return True
        return False
    except Exception:
        return False


def processar_pdf_com_texto(pdf_path: str, nome_arquivo: str) -> Tuple[Optional[List[Document]], Optional[str]]:
    from langchain_community.document_loaders import PyPDFLoader
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        docs = [d for d in docs if d.page_content and d.page_content.strip()]
        if docs:
            for i, d in enumerate(docs):
                d.metadata = {
                    "source": nome_arquivo,
                    "page": i + 1,
                    "file_path": pdf_path,
                    "metodo": "extracao_direta",
                }
            return docs, "extracao_direta"
    except Exception:
        pass
    return None, None


def processar_pdf_com_ocr(
    pdf_path: str,
    nome_arquivo: str,
    pytesseract,
    convert_from_path,
) -> Tuple[Optional[List[Document]], Optional[str]]:
    docs: List[Document] = []
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        total_paginas = len(reader.pages)

        poppler = verificar_poppler()
        for page_num in range(1, total_paginas + 1):
            images = convert_from_path(
                pdf_path,
                dpi=200,
                first_page=page_num,
                last_page=page_num,
                fmt="jpeg",
                poppler_path=poppler if poppler else None,
            )
            if images:
                image = images[0]
                text = pytesseract.image_to_string(image, lang="por", config="--psm 3 --oem 1")
                if text and text.strip():
                    docs.append(
                        Document(
                            page_content=text.strip(),
                            metadata={
                                "source": nome_arquivo,
                                "page": page_num,
                                "file_path": pdf_path,
                                "metodo": "ocr",
                            },
                        )
                    )
    except Exception:
        return None, None
    return (docs, "ocr") if docs else (None, None)


def processar_docx(docx_path: str, nome_arquivo: str) -> Tuple[Optional[List[Document]], Optional[str]]:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(docx_path)
        documents = loader.load()
        documents = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
        if documents:
            for doc in documents:
                doc.metadata = {
                    "source": nome_arquivo,
                    "page": 1,
                    "file_path": docx_path,
                    "metodo": "docx",
                }
            return documents, "docx"
    except Exception:
        pass

    try:
        import docx as docx_lib
        d = docx_lib.Document(docx_path)
        text = "\n".join(p.text for p in d.paragraphs if p.text and p.text.strip())
        if text.strip():
            return [Document(page_content=text.strip(), metadata={
                "source": nome_arquivo,
                "page": 1,
                "file_path": docx_path,
                "metodo": "docx",
            })], "docx"
    except Exception:
        pass

    return None, None
