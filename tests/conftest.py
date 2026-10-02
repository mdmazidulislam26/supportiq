# Shared fixtures: an offline retrieval stack and a RAGService with a fake LLM.
import pytest
from factory import build_stack
from rag import RAGService


@pytest.fixture(scope="session")
def stack():
    return build_stack("offline")


@pytest.fixture()
def service(stack):
    retriever, _ = stack
    from llm import FakeLLM
    return RAGService(retriever, FakeLLM(), threshold=0.05)
