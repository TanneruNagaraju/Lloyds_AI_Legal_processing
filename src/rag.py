"""
Same functionality as current code, but rewritten to read in a clear, top-to-bottom FLOW.

Nothing is removed conceptually — all parts still exist:
- env + auth
- hashing + ingest
- qdrant hybrid store builder
- parent retriever builder (indexing parent->docstore, child->qdrant)
- child-retrieve -> child-rerank -> parent-fetch (custom retriever)
- paraphrase-then-retrieve wrapper
- field query builder + per-field retrieval
- chunk object conversion + dedupe + merge
- build_field_chunks_json_objects aggregator
- a single top-level "run" entrypoint to make usage easy

"""

import os
import re
import hashlib
import torch
import json
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Callable, Literal, get_args
import traceback
from functools import lru_cache
from dotenv import load_dotenv
from huggingface_hub import login

from pydantic import ConfigDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_core.stores import InMemoryStore, BaseStore
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever

from langsmith.wrappers import wrap_openai
from langsmith import traceable

from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import logging

LOGGER = logging.getLogger(__name__)
# ============================================================
# 0) ENV / AUTH
# ============================================================
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)

openai_api_key = os.environ.get("OPENAI_API_KEY")


# ============================================================
# 1) FIELD SPECS (queries + fields list)
# ============================================================
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's question into a short search query for retrieving legal/regulatory text.\n"
            "Rules:\n"
            "- Keep key entities, product terms, dates, and legal action verbs (ban, restrict, tax, labeling, entry into force).\n"
            "- Remove fluff.\n"
            "- Output ONLY the rewritten query.",
        ),
        ("user", "{q}"),
    ]
)



FIELD_QUERIES = {
    "Document Type": {
        "questions": [
            "What type of document is this?",
            "Is this document a lawsuit, notice, court order, settlement offer, regulatory inquiry, coverage letter, or legal correspondence?",
            "How is this document described in its heading or title?",
            "What is the formal category of this document based on the text?",
            "Does the document indicate whether it is a legal notice, court document, claim notice, or regulatory correspondence?",
            "What is the official document label shown at the start of the document?",
        ],
    },
    "Date of Loss": {
        "questions": [
            "What is the date of loss mentioned in this document?",
            "When did the incident, event, accident, or damage occur?",
            "What date is identified as the loss date or incident date?",
            "Does the document specify when the claim event happened?",
            "Is there a stated date of loss, accident date, or incident date?",
            "If a date of loss is not applicable, does the document explicitly say so?",
        ],
    },
    "Policy Number": {
        "questions": [
            "What is the policy number in this document?",
            "Which insurance policy reference is mentioned?",
            "What is the policy identifier associated with this matter?",
            "Does the document specify a policy number or policy reference?",
            "What policy number applies to this claim, notice, or court matter?",
            "Are there multiple policy number variants mentioned, and which one appears most likely to be correct?",
        ],
    },
    "Recipient": {
        "questions": [
            "Who is the recipient of this document?",
            "To whom is this document addressed?",
            "What person, department, organisation, or email address is listed as the recipient?",
            "Does the document specify the addressee or receiving party?",
            "Who is this correspondence intended for?",
            "Is there a named recipient, legal team, claims team, or email address in the header or salutation?",
        ],
    },
    "Claimant": {
        "questions": [
            "Who is the claimant in this document?",
            "Does the document identify a claimant, complainant, plaintiff, or applicant?",
            "Who is bringing the claim or complaint?",
            "What person or entity is named as the claimant?",
            "If this is litigation or settlement correspondence, who is the claimant or plaintiff?",
            "If no claimant is stated, is there another party making the complaint or allegation?",
        ],
    },
    "Defendant": {
        "questions": [
            "Who is the defendant in this document?",
            "Does the document identify a defendant, respondent, insured party, or responding party?",
            "What person or organisation is named as the defendant or respondent?",
            "If this is litigation or settlement correspondence, who is defending the claim?",
            "Who is the opposing party in the matter described?",
            "If the document is not court litigation, is there a named insured, respondent, or target entity instead?",
        ],
    },
    "Case / Court Reference Number": {
        "questions": [
            "What is the case number, court reference number, or regulatory reference number?",
            "Does this document include a case reference, claim reference, filing number, or court identifier?",
            "What reference number is associated with this legal or regulatory matter?",
            "Is there a court case ID, docket number, or inquiry reference stated in the document?",
            "What external reference number should be used to identify this matter?",
            "If no case or court reference exists, does the document explicitly say that none has been assigned?",
        ],
    }

}


FIELDS = [
    "Document Type",
    "Date of Loss",
    "Policy Number",
    "Recipient",
    "Claimant",
    "Defendant",
    "Case / Court Reference Number"
]

# ============================================================
# 2) BASIC UTILITIES (hashing, ingest)
# ============================================================
def _hash(s: str) -> str:
    """
    Create a short, stable hash for a string.

    Args:
        s: Input string.

    Returns:
        A 16-character hex prefix of SHA256(s), used as a compact stable-ish ID.
    """
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def ingest_one_text(
    text: str,
    source: str = "input_text",
    doc_id: Optional[str] = None,
) -> List[Document]:

    """
    Wrap a single large input string into a LangChain Document.

    This is the canonical "ingestion" step for pipeline. It:
    - strips whitespace
    - returns [] for empty text
    - assigns a stable-ish doc_id (hash) if none is provided
    - stores metadata fields rely on later

    Args:
        text: The raw text to ingest.
        source: A label describing the origin of the text (file name, URL, etc.).
        doc_id: Optional explicit document identifier. If None, a hash is used.

    Returns:
        A list with one Document (or empty list if input text is empty).
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    did = doc_id or _hash(source + "::" + cleaned[:2000])
    return [
        Document(
            page_content=cleaned,
            metadata={"source": source, "doc_id": did, "content_type": "text"},
        )
    ]


# ============================================================
# 3) QDRANT HYBRID VECTOR STORE (dense + sparse)
# ============================================================

_SPARSE = FastEmbedSparse(model_name="Qdrant/bm25")


def _embedding_size(embedding_model: str) -> int:
    """
    Return the expected embedding dimensionality for a supported OpenAI embedding model.

    Args:
        embedding_model: Name of the OpenAI embeddings model.

    Returns:
        Integer vector size.

    Raises:
        ValueError: If the embedding model is unknown/unsupported.
    """
    if embedding_model == "text-embedding-3-large":
        return 3072
    if embedding_model == "text-embedding-3-small":
        return 1536
    raise ValueError(f"Unsupported embedding_model: {embedding_model}")


@lru_cache(maxsize=32)
def build_qdrant_hybrid_store(_QDRANT_CLIENT: QdrantClient,
                              collection: str, 
                              embedding_model: str,
                              reset: bool = True) -> QdrantVectorStore:

    """
    Create (or reuse) a Qdrant-backed hybrid vector store (dense + sparse).

    - Ensures the collection exists (creates only if missing).
    - Configures:
      - dense vectors (OpenAI embeddings)
      - sparse vectors (BM25 via FastEmbedSparse)
      - HYBRID retrieval mode

    Args:
        collection: Qdrant collection name.
        embedding_model: OpenAI embedding model name.

    Returns:
        A configured QdrantVectorStore instance.
    """
    embedding_size_ = _embedding_size(embedding_model)

    if reset and _QDRANT_CLIENT.collection_exists(collection):
        _QDRANT_CLIENT.delete_collection(collection)

    # If collection exists, ensure it has BOTH named vectors as expected
    if _QDRANT_CLIENT.collection_exists(collection):
        info = _QDRANT_CLIENT.get_collection(collection)

        # check named dense vector exists with correct size and sparse exists
        has_dense = getattr(info.config.params, "vectors", None) is not None and "dense" in info.config.params.vectors
        has_sparse = getattr(info.config.params, "sparse_vectors", None) is not None and "sparse" in info.config.params.sparse_vectors

        dense_ok = False
        if has_dense:
            dense_ok = info.config.params.vectors["dense"].size == embedding_size_

        if (not has_dense) or (not has_sparse) or (not dense_ok):
            # reset to avoid mixed schema / missing vectors
            _QDRANT_CLIENT.delete_collection(collection)
            _QDRANT_CLIENT.create_collection(
                collection_name=collection,
                vectors_config={"dense": VectorParams(size=embedding_size_, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
            )
    else:
        _QDRANT_CLIENT.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=embedding_size_, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )

  
    dense = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)

    return QdrantVectorStore(
        client=_QDRANT_CLIENT,
        collection_name=collection,
        embedding=dense,
        sparse_embedding=_SPARSE,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


# ============================================================
# 4) PARENT STORE + INDEXING (ParentDocumentRetriever)
#    - used ONLY to create parent->child linkage + docstore population
# ============================================================
_PARENT_DOCSTORE_CACHE: Dict[Tuple[str, str], InMemoryStore] = {}


@lru_cache(maxsize=32)
def build_parent_retriever(
    q_client: QdrantClient,
    collection: str,
    embedding_model: str,
    *,
    parent_chunk_size: int = 500,
    parent_chunk_overlap: int = 30,
    child_chunk_size: int = 100,
    child_chunk_overlap: int = 15,
    k: int = 12,
    parent_id_key: str = "parent_id",
) -> ParentDocumentRetriever:

    """
    Build a ParentDocumentRetriever configured for Qdrant hybrid store.

    This retriever is mainly used for INDEXING in custom design:
    - It stores parent chunks in a docstore
    - It stores child chunks in the vector store, with metadata linking them to parents via parent_id_key

    Note:
        Standard ParentDocumentRetriever retrieval returns parents directly.
        In this design, index with ParentDocumentRetriever, but retrieval is custom
        (child retrieve -> child rerank -> parent fetch).

    Args:
        collection: Qdrant collection name.
        embedding_model: OpenAI embedding model name.
        parent_chunk_size: Chunk size for parent chunks stored in docstore.
        parent_chunk_overlap: Overlap for parent chunks.
        child_chunk_size: Chunk size for child chunks stored in vector store.
        child_chunk_overlap: Overlap for child chunks.
        k: Default number of results when ParentDocumentRetriever is used for retrieval (not critical in custom retrieval path).
        parent_id_key: Metadata key that links a child chunk to its parent ID.

    Returns:
        Configured ParentDocumentRetriever.
    """

    
    vs = build_qdrant_hybrid_store(q_client, collection, embedding_model)

    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
    )
    child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )

    key = (collection, embedding_model)
    if key not in _PARENT_DOCSTORE_CACHE:
        _PARENT_DOCSTORE_CACHE[key] = InMemoryStore()
    docstore = _PARENT_DOCSTORE_CACHE[key]

    pr = ParentDocumentRetriever(
        vectorstore=vs,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs={"k": k},
        id_key=parent_id_key, 
    )

    return pr


@traceable(name="index_one_text_into_qdrant")
def index_one_text_into_qdrant(
    
    text: str,
    client_q: QdrantClient,
    collection: str,
    embedding_model: str,
    source: str = "input_text",
    *,
    parent_chunk_size: int = 500,
    parent_chunk_overlap: int = 30,
    child_chunk_size: int = 100,
    child_chunk_overlap: int = 15,
    k: int = 12,
) -> ParentDocumentRetriever:

    """
    Index one large text into:
    - Qdrant (child chunks; dense+sparse vectors)
    - docstore (parent chunks)

    This function is main "ingestion + indexing" entry for a single text payload.

    Args:
        text: Raw text to index.
        collection: Qdrant collection name.
        embedding_model: OpenAI embedding model.
        source: Metadata "source" label.
        parent_chunk_size: Size of parent chunks stored in docstore.
        parent_chunk_overlap: Overlap for parent chunks.
        child_chunk_size: Size of child chunks embedded into Qdrant.
        child_chunk_overlap: Overlap for child chunks.
        k: Default search-k for ParentDocumentRetriever (not central for custom retrieval).

    Returns:
        The ParentDocumentRetriever instance used for indexing (contains the docstore reference).
    """
    
    docs = ingest_one_text(text=text, source=source)
    if not docs:
        raise ValueError("Empty text provided; nothing to index.")

    pr = build_parent_retriever(
        q_client = client_q,
        collection=collection,
        embedding_model=embedding_model,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        k=k,
    )

    # indexing: parent->docstore, child->qdrant with parent_id metadata
    pr.add_documents(docs)
    return pr


# ============================================================
# 5) PARAPHRASE WRAPPER (unchanged behavior)
# ============================================================

class ParaphraseThenRetrieve(BaseRetriever):
    """
    A retriever wrapper that:
    - rewrites the input query via an LLM (short search-style rewrite)
    - calls an underlying retriever with the rewritten query
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: BaseRetriever
    llm: Any
    prompt: Any

    def _build_rewrite_input(self, query: str):
        if isinstance(self.prompt, str):
            return self.prompt.format(q=query)
        return self.prompt.format_messages(q=query)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Synchronous retrieval:
        - paraphrase/rewrite query
        - retrieve documents using rewritten query
        """
        rewrite_inp = self._build_rewrite_input(query)
        rewritten = self.llm.invoke(rewrite_inp).content.strip()
        if not rewritten:
            rewritten = query
        return self.retriever.invoke(rewritten)
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async retrieval:
        - paraphrase/rewrite query
        - retrieve documents using rewritten query
        """
        rewrite_inp = self._build_rewrite_input(query)
        rewritten = (await self.llm.ainvoke(rewrite_inp)).content.strip()
        if not rewritten:
            rewritten = query
        return await self.retriever.ainvoke(rewritten)


# ============================================================
# 6) CROSS-ENCODER RERANK MODEL (cached)
# ============================================================
@lru_cache(maxsize=1)
def get_rerank_model():
    """
    Load and cache the HuggingFace cross-encoder reranker model.

    Returns:
        A HuggingFaceCrossEncoder instance on GPU if available, else CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3", model_kwargs={"device": device})


# ============================================================
# 7) CUSTOM FLOW RETRIEVER:
#    CHILD retrieve (Qdrant) -> CHILD rerank -> PARENT fetch (docstore)
# ============================================================

class ChildRerankThenParentRetriever(BaseRetriever):

   
    """
    Retrieve with a 3-step “child-first” RAG flow, then return PARENT chunks.

    Overview
    --------
    This retriever implements a custom retrieval strategy designed for
    ParentDocumentRetriever-style indexing, where:

      - CHILD chunks are embedded and stored in a vector store (Qdrant hybrid: dense + sparse)
      - PARENT chunks are stored as raw text in a docstore (InMemoryStore / BaseStore)
      - CHILD chunks contain metadata linking them to a parent via `parent_id_key`

    Retrieval Flow (sync + async)
    -----------------------------
    Given a user query:

      1) CHILD RETRIEVE:
         Run vector search over child chunks using `child_retriever`
         (typically Qdrant hybrid retrieval: dense embeddings + sparse BM25).

      2) CHILD RERANK:
         Re-rank the retrieved child chunks with a cross-encoder reranker
         (`reranker.compress_documents(children, query)`), keeping the most relevant
         child evidence.

      3) PARENT FETCH:
         Extract unique parent IDs from the reranked child chunks (from metadata
         key `parent_id_key`), take the first `parent_top_k` unique IDs, then fetch
         the corresponding parent chunks from `docstore` via `mget`.

    What It Returns
    ---------------
    Returns a list of PARENT `Document` objects (raw parent chunk texts) whose
    ordering is driven by the ranked CHILD matches.

    Key Notes
    ---------
    - Parent chunks are NOT embedded in this design.
      Only child chunks live in the vector store; parents are fetched by ID.
    - This is useful when we want child-level retrieval precision
      (smaller chunks) but parent-level context for downstream tasks
      (bigger chunks for extraction/summarization).

    Attributes
    ----------
    child_retriever : BaseRetriever
        Retriever over CHILD chunks (usually created from a vector store like
        `QdrantVectorStore.as_retriever()`).
    docstore : BaseStore
        Store containing PARENT chunks, keyed by parent IDs (e.g., InMemoryStore).
    reranker : Any
        Cross-encoder reranker/compressor that implements
        `compress_documents(docs, query) -> List[Document]`.
    parent_id_key : str, default="parent_id"
        Metadata key on CHILD documents that contains the parent chunk ID.
    parent_top_k : int, default=2
        Maximum number of unique parent chunks to return after mapping from
        reranked children.

    Returns
    -------
    List[Document]
        Parent `Document` chunks fetched from `docstore` corresponding to the
        highest-scoring reranked child matches.

    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    child_retriever: BaseRetriever
    docstore: BaseStore
    reranker: Any
    parent_id_key: str = "parent_id"
    parent_top_k: int = 2

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Synchronous retrieval:
        - vector search children
        - cross-encoder rerank children
        - fetch parent docs
        """
        children = self.child_retriever.invoke(query)
        if not children:
            return []

        reranked_children = self.reranker.compress_documents(children, query)

        parent_ids: List[str] = []
        seen = set()
        for ch in reranked_children:
            pid = (ch.metadata or {}).get(self.parent_id_key)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            parent_ids.append(pid)
            if len(parent_ids) >= self.parent_top_k:
                break

        if not parent_ids:
            return []

        parents = self.docstore.mget(parent_ids)
        return [p for p in parents if p is not None]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async retrieval:
        - async vector search children
        - rerank in a background thread (reranker is typically sync)
        - fetch parents from docstore
        """
        children = await self.child_retriever.ainvoke(query)
        if not children:
            return []

        reranked_children = await asyncio.to_thread(self.reranker.compress_documents, children, query)

        parent_ids: List[str] = []
        seen = set()
        for ch in reranked_children:
            pid = (ch.metadata or {}).get(self.parent_id_key)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            parent_ids.append(pid)
            if len(parent_ids) >= self.parent_top_k:
                break

        if not parent_ids:
            return []

        parents = await asyncio.to_thread(self.docstore.mget, parent_ids)
        return [p for p in parents if p is not None]


# ============================================================
# 8) BUILD ADVANCED RETRIEVER ("one thing to call")
# ============================================================
@traceable(name="build_advanced_retriever")
def build_advanced_retriever(
    parent_retriever: ParentDocumentRetriever,
    vs: QdrantVectorStore,
    *,
    llm: Optional[Any]=None,
    k: int = 6,            # child candidates
    top_n: int = 5,         # reranked children kept
    parent_top_k: int = 2,  # parents returned
):

    """
    Build the end-to-end retriever used by field extraction pipeline.

    Flow:
        user query -> paraphrase -> child retrieve (Qdrant) -> child rerank -> parent fetch

    Args:
        llm: LLM used for query rewriting (e.g., ChatOpenAI instance).
        parent_retriever: The ParentDocumentRetriever used for indexing; provides access to the docstore.
        vs: QdrantVectorStore used to create the child retriever.
        k: Number of child candidates retrieved from vector store.
        top_n: Number of top child chunks to keep after reranking.
        parent_top_k: Max number of parent documents to return after mapping/deduplication.

    Returns:
        A BaseRetriever can pass into existing pipeline (retrieve_for_field, etc.).
    """
    # child retrieval from qdrant
    child_retriever = vs.as_retriever(search_kwargs={"k": k})

    # rerank children
    child_reranker = CrossEncoderReranker(model=get_rerank_model(), top_n=top_n)

    # map children->parents using the SAME docstore as ParentDocumentRetriever
    child_then_parent = ChildRerankThenParentRetriever(
        child_retriever=child_retriever,
        docstore=parent_retriever.docstore,
        reranker=child_reranker,
        parent_id_key=getattr(parent_retriever, "id_key", "parent_id"),
        parent_top_k=parent_top_k,
    )
    if llm:
        # paraphrase wrapper remains exactly the same
        LOGGER.info("ParaphraseThenRetrieve with OpenAI")
        return ParaphraseThenRetrieve(retriever=child_then_parent, llm=llm, prompt=REWRITE_PROMPT)
    else:
        LOGGER.info("No paraphrasing (direct retrieval)")
        return child_then_parent


# ============================================================
# 9) FIELD QUERY BUILDING + RETRIEVAL
# ============================================================

@traceable(name="retrieve_chunks_for_field_all_questions")
def retrieve_chunks_for_field_all_questions(
    adv_retriever,
    field_name: str,
    top_n_chunks_per_question: int = 5,
) -> List[Document]:
    """
    Retrieve docs for ALL question variants of a field,
    convert to chunk objects, and return ONE merged (deduped) chunk list.
    """
    spec = FIELD_QUERIES[field_name]
    questions = spec.get("questions", [])
    all_docs: List[Document] = []

    for q in questions:
        docs = adv_retriever.invoke(q)[:top_n_chunks_per_question]
        all_docs.extend(docs)

    return all_docs

    
# ============================================================
# 10) DOCS -> CHUNK OBJECTS + DEDUPE + MERGE
# ============================================================
def docs_to_chunk_objects(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Convert retrieved Documents to lightweight dict objects.

    This allows to:
    - store rank
    - keep metadata fields care about
    - operate on text + metadata uniformly in downstream utilities

    Args:
        docs: Retrieved Documents.

    Returns:
        List of dict objects with rank/text + selected metadata keys.
    """
    out: List[Dict[str, Any]] = []
    for rank, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        text = (d.page_content or "").strip()
        out.append(
            {
                "rank": rank,
                "text": text,
                "source": meta.get("source"),
                "doc_id": meta.get("doc_id"),
                "page": meta.get("page"),
                "content_type": meta.get("content_type"),
            }
        )
    return out


def merge_chunk_texts(chunks: List[Dict[str, Any]], sep: str = "\n") -> str:
    """
    Merge chunk dict objects into one long text string.

    Args:
        chunks: List of chunk dicts each containing "text".
        sep: Separator between chunks.

    Returns:
        Concatenated text for downstream prompting/extraction.
    """
    texts = [(c.get("text") or "").strip() for c in chunks]
    texts = [t for t in texts if t]
    return sep.join(texts).strip()


def _normalize_text_for_dedupe(text: str) -> str:
    """
    Normalize text so that insignificant whitespace differences do not produce duplicates.

    Args:
        text: Raw chunk text.

    Returns:
        Normalized text (trimmed, collapsed whitespace).
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def dedupe_chunks_preserve_order(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate chunk dicts while preserving their original order.

    Deduplication key strategy:
    - Prefer (doc_id, page, normalized_text) when available.
    - Else (source, page, normalized_text)
    - Else (normalized_text) only.

    Args:
        chunks: Chunk dict list.

    Returns:
        Deduplicated chunk dict list.
    """
    seen = set()
    out: List[Dict[str, Any]] = []

    for c in chunks:
        text_norm = _normalize_text_for_dedupe(c.get("text", ""))

        if c.get("doc_id") is not None and c.get("page") is not None:
            key = (c.get("doc_id"), c.get("page"), text_norm)
        elif c.get("source") is not None and c.get("page") is not None:
            key = (c.get("source"), c.get("page"), text_norm)
        else:
            key = (text_norm,)

        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    return out


# ============================================================
# 11) MAIN AGGREGATOR OVER ALL FIELDS
# ============================================================
def build_field_chunks_json_objects(
    adv_retriever,
    top_n_chunks: int = 5,
    merge_sep: str = "\n",
    field_sep: str = "\n",
) -> str:
    
    """
    For each field in FIELDS:
    - retrieve top-N documents
    - convert to chunk dicts
    - dedupe
    - merge into a field-specific merged text block
    Finally merge all fields into one combined string.

    Args:
        adv_retriever: advanced retriever (output of build_advanced_retriever).
        top_n_chunks: How many docs/chunks to take per field (after retrieval).
        merge_sep: Separator when merging chunks within a field.
        field_sep: Separator when merging fields.

    Returns:
        A single merged string containing merged evidence across all fields.
    """
    parts: List[str] = []
    for field in FIELDS:
   
        LOGGER.info("Field : %s", field)
        docs = retrieve_chunks_for_field_all_questions(adv_retriever, field, top_n_chunks)
        chunks = docs_to_chunk_objects(docs)
        chunks = dedupe_chunks_preserve_order(chunks)
        

        LOGGER.info("Unique chunks : %s", len(chunks))
        merged_text = merge_chunk_texts(chunks, sep=merge_sep)
        if merged_text.strip():
            parts.append(f"{merged_text}")
    return field_sep.join(parts)


# ============================================================
# 12) ONE SIMPLE ENTRYPOINT 
# ============================================================
@traceable(name="run_rag_fields")
def run_rag_fields(
    *,
    text: str,
    collection: str,
    embedding_model: str,
    llm: Optional[Any]= None,
    source: str = "input_text",
    # indexing split sizes
    parent_chunk_size: int = 300,
    parent_chunk_overlap: int = 30,
    child_chunk_size: int = 50,
    child_chunk_overlap: int = 8,
    # retrieval controls
    child_k: int = 10,
    rerank_top_n: int = 8,
    parent_top_k: int = 5,
    field_top_n_chunks: int = 5,
) -> str:

    """
    End-to-end helper to run pipeline in a single call.

    Steps:
        1) Index text (parent->docstore, child->qdrant)
        2) Build advanced retriever:
           paraphrase -> child retrieve -> child rerank -> parent fetch
        3) For each field in FIELDS, retrieve evidence and merge into one string

    Args:
        text: The raw text to index and query over.
        collection: Qdrant collection name.
        embedding_model: OpenAI embeddings model name.
        llm: LLM instance used for query rewriting (ChatOpenAI, etc.).
        source: Metadata label for the document source.
        parent_chunk_size: Parent chunk size stored in docstore.
        parent_chunk_overlap: Parent overlap.
        child_chunk_size: Child chunk size stored in Qdrant.
        child_chunk_overlap: Child overlap.
        child_k: Number of child candidates retrieved from Qdrant per query.
        rerank_top_n: Number of child chunks kept after cross-encoder reranking.
        parent_top_k: Maximum number of parents returned after mapping children->parents.
        field_top_n_chunks: How many returned parent docs to use per field.

    Returns:
        One merged evidence string across all fields.
    """

    # clear cached store/retriever objects for a clean run
    build_qdrant_hybrid_store.cache_clear()
    build_parent_retriever.cache_clear()

    client_ = QdrantClient(":memory:")   # <-- fresh per call

    # reset collection ONCE (clean schema)
    _ = build_qdrant_hybrid_store(client_, collection, embedding_model, reset=True)
    
    # 1) index (creates child vectors + parent docstore)
    parent_retriever = index_one_text_into_qdrant(
        text=text,
        client_q=client_,
        collection=collection,
        embedding_model=embedding_model,
        source=source,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )

    LOGGER.info("created child vectors + parent docstore")

    # 2) vectorstore for child retrieval

    vs = build_qdrant_hybrid_store(client_, collection, embedding_model, reset=False)

    LOGGER.info("created vectorstore for child retrieval")

    # 3) advanced retriever (paraphrase -> child retrieve -> child rerank -> parent fetch)
    adv = build_advanced_retriever(
        parent_retriever=parent_retriever,
        vs=vs,
        llm=llm,
        k=child_k,
        top_n=rerank_top_n,
        parent_top_k=parent_top_k,
    )

    
    LOGGER.info("paraphrase -> child retrieve -> child rerank -> parent fetch")
    # 4) build merged context across fields
    return build_field_chunks_json_objects(adv,field_top_n_chunks)
