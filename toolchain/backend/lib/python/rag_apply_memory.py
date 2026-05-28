def get_documents(topk_ids, corpus):
    documents = []
    for i in range(len(topk_ids)):
        idx = int(topk_ids[i])
        if 0 <= idx < len(corpus):
            retrieved_docs.append(corpus[idx])
        else:
            logger.warning(f"Invalid document index {idx}, skipping")
    
    return documents

def _has_chat_template(tokenizer) -> bool:
    """Check if the tokenizer has a chat template."""
    if tokenizer is None:
        return False
    # Check if chat_template attribute exists and is not None/empty
    return (hasattr(tokenizer, 'chat_template') and 
            tokenizer.chat_template is not None and
            len(tokenizer.chat_template) > 0)

def _build_context_string(documents: List[Tuple[str, float]], 
                            context_prefix: str = "") -> str:
    """Build context string from retrieved documents."""
    context_parts = []
    for i, (doc, score) in enumerate(documents, 1):
        # Truncate long documents
        doc_text = doc[:1000] if len(doc) > 1000 else doc
        context_parts.append(f"[Document {i}] {doc_text}")
    
    context = "\n\n".join(context_parts)
    return f"{context_prefix}{context}" if context_prefix else context

def _build_rag_prompt(tokenizer, query: str, documents: List[Tuple[str, float]], 
                        context_prefix: str = "") -> str:
    """Build a RAG prompt with retrieved documents, using chat template if available."""
    context = _build_context_string(documents, context_prefix)
    
    # Build the user message content
    user_content = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}"""

    # Check if model has a chat template
    if _has_chat_template(tokenizer):
        logger.info("Using model's chat template for prompt formatting")
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
            {"role": "user", "content": user_content}
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using default format.")
    
    # Fallback to default format (no chat template)
    logger.info("Using default prompt format (no chat template available)")
    prompt = f"""{user_content}

Answer:"""
    
    return prompt
    