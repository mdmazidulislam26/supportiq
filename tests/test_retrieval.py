# Tests for chunking, rank fusion and hybrid retrieval.
from corpus import DOCS
from retrieval import chunk_documents, rrf_fuse, tokenize


# Every document is chunked and chunk ids are unique.
def test_chunk_ids_unique_and_every_doc_covered():
    chunks = chunk_documents(DOCS)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.doc_id for c in chunks} == {d["id"] for d in DOCS}


# Chunks end on sentence boundaries, so facts are never cut in half.
def test_chunks_never_cut_sentences():
    for c in chunk_documents(DOCS):
        assert c.text.rstrip().endswith((".", "!", "?"))


# Items ranked high by both retrievers win after fusion.
def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = dict(rrf_fuse([[1, 2, 3], [3, 1, 4]]))
    assert fused[1] > fused[2] and fused[1] > fused[4]


# Obvious queries retrieve the right document.
def test_hybrid_finds_obvious_documents(stack):
    retriever, _ = stack
    chunks, _ = retriever.search("password reset link valid", k=4)
    assert "accounts" in [c.doc_id for c in chunks]
    chunks, _ = retriever.search("which wallets accepted bKash Nagad", k=4)
    assert "payment" in [c.doc_id for c in chunks]


# Tokenizer keeps numbers and % intact.
def test_tokenizer_keeps_percent_and_numbers():
    assert tokenize("Get 20% off in 14 days") == ["get", "20%", "off", "in", "14", "days"]
