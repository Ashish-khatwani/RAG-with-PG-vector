SYSTEM_PROMPT = """You are a document intelligence assistant.
Answer only from the provided context.
If the answer is not supported by the context, say that the documents do not contain enough information.
Be precise, concise, and cite source file names with chunk numbers inline when useful."""


def build_qa_prompt(question: str, context_blocks: list[str]) -> str:
    joined_context = "\n\n".join(context_blocks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{joined_context}\n\n"
        "Answer:"
    )
