def format_prompt(query: str, context_chunks: list[str], source_docs: list[str]) -> str:
    if not context_chunks:
        return "sorry i dont have any information regarding that."

    context_str = "\n".join(f"- {chunk}" for chunk in context_chunks)
    
    prompt = f"""You are a strict, helpful AI assistant.
Answer the User Query STRICTLY AND ONLY using the provided Context.
If the Context does not contain the answer to the User Query (e.g. if the user asks an unrelated question like which game has spiderman in it), you MUST reply EXACTLY with:
sorry i dont have any information regarding that.
Do not add any other text, apologies, or explanations. Do not use outside knowledge.

Context:
{context_str}

User Query: {query}"""
    return prompt
