import os
import json
import base64
import glob
import hashlib
import shutil
from collections import defaultdict
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_cohere import CohereRerank

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

load_dotenv()

DOCS_PATH = "docs"
PERSIST_DIRECTORY = "db/chroma_combined"
TEXT_EXTENSIONS = {".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PDF_EXTENSIONS = {".pdf"}

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)


class QueryVariations(BaseModel):
    queries: List[str]


# ingestion

def partition_pdf_file(file_path):
    """Extract text/table/image-aware chunks from a PDF via unstructured"""
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    normalized = []
    for chunk in chunks:
        tables, images = [], []
        if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__
                if element_type == "Table":
                    tables.append(getattr(element.metadata, "text_as_html", element.text))
                elif element_type == "Image" and hasattr(element.metadata, "image_base64"):
                    images.append(element.metadata.image_base64)
        normalized.append({"text": chunk.text, "tables": tables, "images": images, "source": file_path})
    return normalized


def partition_text_file(file_path, chunk_size=1000, chunk_overlap=200):
    """Split a plain text file into text-only chunks"""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pieces = splitter.split_text(raw_text)
    return [{"text": piece, "tables": [], "images": [], "source": file_path} for piece in pieces]


def partition_image_file(file_path):
    """Treat a standalone image file as a single image-only chunk"""
    with open(file_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    return [{"text": "", "tables": [], "images": [image_base64], "source": file_path}]


def partition_all_documents(docs_path):
    """Walk docs_path and normalize every supported file into chunks"""
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    all_chunks = []
    for file_path in sorted(glob.glob(os.path.join(docs_path, "*"))):
        ext = os.path.splitext(file_path)[1].lower()
        print(f"Partitioning {file_path}...")
        if ext in TEXT_EXTENSIONS:
            all_chunks.extend(partition_text_file(file_path))
        elif ext in PDF_EXTENSIONS:
            all_chunks.extend(partition_pdf_file(file_path))
        elif ext in IMAGE_EXTENSIONS:
            all_chunks.extend(partition_image_file(file_path))
        else:
            print(f"  Skipping unsupported file type: {file_path}")

    if not all_chunks:
        raise FileNotFoundError(
            f"No supported documents (.txt, .pdf, .png, .jpg, .jpeg) found in {docs_path}."
        )

    print(f"Partitioned {len(all_chunks)} chunks from documents in {docs_path}")
    return all_chunks


def create_ai_enhanced_summary(text, tables, images, source):
    """Use GPT-4o vision to create a rich, searchable description for mixed content"""
    try:
        prompt_text = f"""You are creating a searchable description for document content retrieval.

Source file: {source}

TEXT CONTENT:
{text}

"""
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
YOUR TASK:
Generate a comprehensive, searchable description that covers:
1. Key facts, numbers, and data points from text and tables
2. Main topics and concepts discussed
3. Questions this content could answer
4. Visual content analysis (charts, diagrams, patterns in images)
5. Alternative search terms users might use

Make it detailed and searchable - prioritize findability over brevity.

SEARCHABLE DESCRIPTION:"""

        message_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            })

        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
    except Exception as e:
        print(f"  AI summary failed for {source}: {e}")
        summary = f"{text[:300]}..." if text else f"[Image from {source}]"
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarize_chunks(raw_chunks):
    """Turn normalized chunks into LangChain Documents with searchable page_content"""
    documents = []
    total = len(raw_chunks)
    for i, chunk in enumerate(raw_chunks, 1):
        print(f"Summarizing chunk {i}/{total} ({chunk['source']})...")
        if chunk["tables"] or chunk["images"]:
            searchable_content = create_ai_enhanced_summary(
                chunk["text"], chunk["tables"], chunk["images"], chunk["source"]
            )
        else:
            searchable_content = chunk["text"]

        documents.append(Document(
            page_content=searchable_content,
            metadata={
                "source": chunk["source"],
                "original_content": json.dumps({
                    "raw_text": chunk["text"],
                    "tables_html": chunk["tables"],
                    "images_base64": chunk["images"],
                }),
            },
        ))
    return documents


# vector store


def manifest_path(persist_directory):
    return f"{persist_directory}.manifest.json"


def compute_docs_signature(docs_path):
    """Hash every file's path/content so we can detect added/removed/edited docs.

    Content-based (not mtime-based) so the signature is stable across git
    clones/deploys, where checkout mtimes don't match the original ingestion run.
    """
    overall = hashlib.sha256()
    for file_path in sorted(glob.glob(os.path.join(docs_path, "*"))):
        overall.update(file_path.encode("utf-8"))
        with open(file_path, "rb") as f:
            overall.update(hashlib.sha256(f.read()).digest())
    return overall.hexdigest()


def read_manifest_signature(persist_directory):
    path = manifest_path(persist_directory)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f).get("signature")


def write_manifest_signature(persist_directory, signature):
    with open(manifest_path(persist_directory), "w") as f:
        json.dump({"signature": signature}, f)


def build_or_load_vectorstore(docs_path, persist_directory):
    current_signature = compute_docs_signature(docs_path)
    stored_signature = read_manifest_signature(persist_directory)

    if os.path.exists(persist_directory) and current_signature == stored_signature:
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return vectorstore

    if os.path.exists(persist_directory):
        print(f"{docs_path} has changed since last ingestion. Rebuilding vector store...")
        shutil.rmtree(persist_directory)
    else:
        print("No existing vector store found. Running full ingestion pipeline...")

    raw_chunks = partition_all_documents(docs_path)
    documents = summarize_chunks(raw_chunks)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    write_manifest_signature(persist_directory, current_signature)
    print(f"Vector store created with {len(documents)} chunks at {persist_directory}")
    return vectorstore


def load_all_documents(vectorstore):
    """Reconstruct Document objects from the vector store, needed for BM25"""
    raw = vectorstore.get()
    return [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(raw["documents"], raw["metadatas"])
    ]


def build_hybrid_retriever(vectorstore, documents, k=10):
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.7, 0.3])


# accuracy pipline for history-aware rewrite -> multi-query -> RRF -> rerank

def rewrite_standalone_query(user_question, chat_history):
    if not chat_history:
        return user_question
    messages = [
        SystemMessage(content=(
            "Given the chat history, rewrite the new question to be a standalone, "
            "searchable query. Just return the rewritten question."
        )),
    ] + chat_history + [HumanMessage(content=f"New question: {user_question}")]
    result = llm.invoke(messages)
    return result.content.strip()


def generate_query_variations(standalone_query, n=3):
    llm_with_structure = llm.with_structured_output(QueryVariations)
    prompt = f"""Generate {n} different variations of this query that would help retrieve relevant documents:

Original query: {standalone_query}

Return {n} alternative queries that rephrase or approach the same question from different angles."""
    response = llm_with_structure.invoke(prompt)
    return response.queries


def reciprocal_rank_fusion(chunk_lists, k=60):
    rrf_scores = defaultdict(float)
    unique_chunks = {}
    for chunks in chunk_lists:
        for position, chunk in enumerate(chunks, 1):
            content = chunk.page_content
            unique_chunks[content] = chunk
            rrf_scores[content] += 1 / (k+position)

    return sorted(
        ((unique_chunks[content], score) for content, score in rrf_scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )

_reranker = None
_rerank_unavailable = False

def get_reranker(top_n):
    """Build the Cohere reranker once; fall back permanently if no CO_API_KEY is set"""
    global _reranker, _rerank_unavailable
    if _rerank_unavailable:
        return None
    if _reranker is None:
        try:
            _reranker = CohereRerank(model="rerank-english-v3.0", top_n=top_n)
        except Exception as e:
            print(f"Cohere reranker unavailable ({e}). Falling back to RRF ranking only.")
            _rerank_unavailable = True
            return None
    return _reranker


def rerank_candidates(candidates, query, top_n=5):
    reranker = get_reranker(top_n)
    if reranker is None:
        return candidates[:top_n]
    try:
        return reranker.compress_documents(candidates, query)
    except Exception as e:
        print(f"Reranking failed ({e}). Falling back to RRF ranking only.")
        return candidates[:top_n]


# answer generation


def generate_answer(top_docs, user_question, chat_history):
    prompt_text = f"""Based on the following documents from multiple sources, answer this question directly and concisely: {user_question}

If the answer requires combining information from multiple documents or subjects, synthesize them clearly. If the documents don't contain enough information, say "I don't have enough information to answer that question based on the provided documents."

CONTENT TO ANALYZE:
"""
    image_blocks = []
    for i, doc in enumerate(top_docs, 1):
        prompt_text += f"\n--- Document {i} (source: {doc.metadata.get('source', 'unknown')}) ---\n"
        original = json.loads(doc.metadata.get("original_content", "{}"))

        raw_text = original.get("raw_text", "")
        if raw_text:
            prompt_text += f"TEXT:\n{raw_text}\n"

        for j, table in enumerate(original.get("tables_html", []), 1):
            prompt_text += f"TABLE {j}:\n{table}\n"

        for image_base64 in original.get("images_base64", []):
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            })

    prompt_text += "\nANSWER:"
    message_content = [{"type": "text", "text": prompt_text}] + image_blocks

    messages = [
        SystemMessage(content=(
            "You are a direct, helpful assistant that answers using only the provided "
            "documents and conversation history."
        )),
    ] + chat_history + [HumanMessage(content=message_content)]

    result = llm.invoke(messages)
    print(f"\n{result.content}")
    return result.content


# CHAT :3

def ask_question(user_question, hybrid_retriever, chat_history, fetch_k=15, final_k=5):
    

    standalone_query = rewrite_standalone_query(user_question, chat_history)

    variations = generate_query_variations(standalone_query)
    all_queries = [standalone_query] + variations

    all_results = [hybrid_retriever.invoke(q) for q in all_queries]

    fused = reciprocal_rank_fusion(all_results, k=60)
    candidates = [doc for doc, _ in fused[:fetch_k]]

    top_docs = rerank_candidates(candidates, standalone_query, top_n=final_k)

    answer = generate_answer(top_docs, user_question, chat_history)

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    return answer


def start_chat():
    vectorstore = build_or_load_vectorstore(DOCS_PATH, PERSIST_DIRECTORY)
    documents = load_all_documents(vectorstore)
    hybrid_retriever = build_hybrid_retriever(vectorstore, documents)

    chat_history = []
    print("\nAsk me anything about food waste!")
    while True:
        question = input("\nYour question: ")
        if question.lower() == "quit":
            print("Goodbye!")
            break
        ask_question(question, hybrid_retriever, chat_history)


if __name__ == "__main__":
    start_chat()
