def format_prompt(query: str, context_chunks: list[str], source_docs: list[str]) -> str:
    context_str = "\n".join(f"- {chunk}" for chunk in context_chunks)
    sources_str = ", ".join(source_docs)
    
    prompt = f"""You are an intelligent assistant. Use the following context to answer the query.
If the context does not contain enough information to answer the query, say 'I don't have enough information'.
Cite your sources based on the provided documents.

Context:
{context_str}

Sources: {sources_str}

Query: {query}

Answer:"""
    return prompt
