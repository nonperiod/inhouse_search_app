"""data フォルダ内のファイルを読み込み、Chroma ベクトルストアを構築するスクリプト。

使い方:
    python ingest.py
"""
from __future__ import annotations

import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBEDDING_MODEL, PERSIST_DIR

LOADER_MAP = {
    ".pdf": lambda path: PyPDFLoader(path),
    ".docx": lambda path: Docx2txtLoader(path),
    ".csv": lambda path: CSVLoader(path, encoding="utf-8"),
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
}


def load_documents():
    documents = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        loader_factory = LOADER_MAP.get(path.suffix.lower())
        if loader_factory is None:
            continue

        relative = path.relative_to(DATA_DIR)
        try:
            docs = loader_factory(str(path)).load()
        except Exception as exc:  # noqa: BLE001
            print(f"読み込み失敗: {relative} ({exc})")
            continue

        category = relative.parts[0] if len(relative.parts) > 1 else "その他"
        for doc in docs:
            doc.metadata["source"] = str(relative)
            doc.metadata["category"] = category
            if category == "MTG議事録":
                doc.metadata["subcategory"] = relative.parts[1] if len(relative.parts) > 2 else "その他"

        documents.extend(docs)
        print(f"読み込み完了: {relative} ({len(docs)} ページ/レコード)")

    return documents


def build_vectorstore():
    load_dotenv()

    documents = load_documents()
    if not documents:
        raise SystemExit("data フォルダに対応ファイル(pdf/docx/csv/txt)が見つかりませんでした。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"チャンク数: {len(chunks)}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    print(f"ベクトルストアを作成しました: {PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
