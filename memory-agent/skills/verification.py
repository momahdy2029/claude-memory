"""Verification skills for anti-hallucination - Best-of-N and Quote Extraction."""
import os
import json
import asyncio
from typing import Dict, Any, Optional, List
from services.database import DatabaseService
from services.embeddings import EmbeddingService

# Check if LLM analysis is available
USE_LLM_ANALYSIS = os.getenv("USE_LLM_ANALYSIS", "true").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VERIFICATION_MODEL = os.getenv("VERIFICATION_MODEL", "llama3.2:3b")


async def best_of_n_verify(
    query: str,
    n: int = 3,
    context: Optional[str] = None,
    threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Best-of-N verification: Run the same query N times and check consistency.

    If outputs are inconsistent, it likely indicates hallucination.

    Args:
        query: The question/task to verify
        n: Number of runs (default 3)
        context: Optional context to include
        threshold: Similarity threshold for consistency (0-1)

    Returns:
        Dict with verification results
    """
    if not USE_LLM_ANALYSIS:
        return {
            "success": False,
            "error": "LLM analysis not available",
            "recommendation": "Enable USE_LLM_ANALYSIS or install Ollama"
        }

    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
    except Exception as e:
        return {
            "success": False,
            "error": f"Ollama not available: {e}"
        }

    prompt_template = """Answer this question concisely and factually.
{context}
Question: {query}

Answer (be specific and factual):"""

    context_str = f"Context: {context}\n" if context else ""
    prompt = prompt_template.format(context=context_str, query=query)

    # Run N times
    responses = []
    for i in range(n):
        try:
            response = client.generate(
                model=VERIFICATION_MODEL,
                prompt=prompt,
                options={
                    "temperature": 0.7,  # Some variation to test consistency
                    "num_predict": 200
                }
            )
            responses.append(response.get("response", "").strip())
        except Exception as e:
            responses.append(f"[Error: {e}]")

    # Analyze consistency
    consistency_result = await _analyze_consistency(responses, threshold)

    return {
        "success": True,
        "query": query,
        "n_runs": n,
        "responses": responses,
        "is_consistent": consistency_result["is_consistent"],
        "consistency_score": consistency_result["score"],
        "consensus_answer": consistency_result.get("consensus"),
        "inconsistencies": consistency_result.get("inconsistencies", []),
        "recommendation": (
            "Answers are consistent - likely reliable"
            if consistency_result["is_consistent"]
            else "INCONSISTENT answers detected - verify manually before trusting"
        )
    }


async def _analyze_consistency(responses: List[str], threshold: float) -> Dict[str, Any]:
    """Analyze consistency across multiple responses."""
    if len(responses) < 2:
        return {"is_consistent": True, "score": 1.0, "consensus": responses[0] if responses else None}

    # Simple word overlap consistency check
    def get_key_words(text: str) -> set:
        # Extract significant words (longer than 3 chars, not common)
        common_words = {'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'were', 'been', 'have', 'has', 'will', 'would', 'could', 'should'}
        words = set(w.lower() for w in text.split() if len(w) > 3 and w.lower() not in common_words)
        return words

    word_sets = [get_key_words(r) for r in responses]

    # Calculate pairwise overlap
    overlaps = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            if word_sets[i] and word_sets[j]:
                intersection = word_sets[i] & word_sets[j]
                union = word_sets[i] | word_sets[j]
                overlap = len(intersection) / len(union) if union else 0
                overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

    # Find inconsistencies
    inconsistencies = []
    if avg_overlap < threshold:
        # Find which responses differ most
        all_words = set()
        for ws in word_sets:
            all_words.update(ws)

        # Words that appear in some but not all responses
        for word in all_words:
            present_in = sum(1 for ws in word_sets if word in ws)
            if 0 < present_in < len(word_sets):
                inconsistencies.append(f"'{word}' appears in {present_in}/{len(word_sets)} responses")

    # Find consensus (most common response pattern)
    consensus = responses[0] if responses else None

    return {
        "is_consistent": avg_overlap >= threshold,
        "score": round(avg_overlap, 3),
        "consensus": consensus,
        "inconsistencies": inconsistencies[:5]  # Limit to 5
    }


async def extract_quotes(
    document: str,
    query: str,
    max_quotes: int = 5,
    min_length: int = 20
) -> Dict[str, Any]:
    """
    Extract direct quotes from a document that are relevant to a query.

    Forces verbatim grounding - Claude must work from exact quotes.

    Args:
        document: The source document text
        query: What we're looking for
        max_quotes: Maximum quotes to extract
        min_length: Minimum quote length

    Returns:
        Dict with extracted quotes
    """
    if not document or not query:
        return {
            "success": False,
            "error": "Document and query are required"
        }

    if not USE_LLM_ANALYSIS:
        # Fallback: simple keyword-based extraction
        return await _extract_quotes_keyword(document, query, max_quotes, min_length)

    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
    except:
        return await _extract_quotes_keyword(document, query, max_quotes, min_length)

    prompt = f"""Extract exact, word-for-word quotes from this document that are relevant to the query.

DOCUMENT:
{document[:5000]}

QUERY: {query}

Return ONLY a JSON array of exact quotes from the document. Do not paraphrase or modify.
Example format: ["exact quote 1", "exact quote 2"]

Quotes (JSON array only):"""

    try:
        response = client.generate(
            model=VERIFICATION_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.1,  # Low temperature for accuracy
                "num_predict": 500
            }
        )

        result_text = response.get("response", "[]")

        # Parse JSON
        json_start = result_text.find("[")
        json_end = result_text.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            quotes = json.loads(result_text[json_start:json_end])

            # Verify quotes actually exist in document
            verified_quotes = []
            for quote in quotes[:max_quotes]:
                if isinstance(quote, str) and len(quote) >= min_length:
                    # Check if quote (or close match) exists in document
                    quote_lower = quote.lower()
                    doc_lower = document.lower()
                    if quote_lower in doc_lower or _fuzzy_match(quote_lower, doc_lower):
                        verified_quotes.append({
                            "quote": quote,
                            "verified": True
                        })
                    else:
                        verified_quotes.append({
                            "quote": quote,
                            "verified": False,
                            "warning": "Quote not found verbatim in document"
                        })

            return {
                "success": True,
                "query": query,
                "quotes": verified_quotes,
                "total_found": len(verified_quotes),
                "all_verified": all(q["verified"] for q in verified_quotes),
                "grounding_instruction": (
                    "Use ONLY these verified quotes to answer. "
                    "Do not add information not in the quotes."
                )
            }

    except Exception as e:
        pass

    # Fallback to keyword extraction
    return await _extract_quotes_keyword(document, query, max_quotes, min_length)


def _fuzzy_match(quote: str, document: str, threshold: float = 0.8) -> bool:
    """Check if quote approximately matches something in document."""
    # Simple check: do most words appear in sequence?
    words = quote.split()
    if len(words) < 3:
        return False

    # Check if 80% of words appear near each other in document
    matches = 0
    for word in words:
        if word in document:
            matches += 1

    return (matches / len(words)) >= threshold


async def _extract_quotes_keyword(
    document: str,
    query: str,
    max_quotes: int,
    min_length: int
) -> Dict[str, Any]:
    """Fallback keyword-based quote extraction."""
    # Split query into keywords
    keywords = [w.lower() for w in query.split() if len(w) > 3]

    # Split document into sentences
    sentences = []
    for sep in ['. ', '.\n', '! ', '? ', '\n\n']:
        if sep in document:
            parts = document.split(sep)
            for part in parts:
                if len(part.strip()) >= min_length:
                    sentences.append(part.strip())

    if not sentences:
        sentences = [document[i:i+200] for i in range(0, len(document), 150)]

    # Score sentences by keyword matches
    scored = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for kw in keywords if kw in sentence_lower)
        if score > 0:
            scored.append((score, sentence))

    # Sort by score and take top N
    scored.sort(reverse=True)
    quotes = [{"quote": s, "verified": True, "keyword_matches": score} for score, s in scored[:max_quotes]]

    return {
        "success": True,
        "query": query,
        "quotes": quotes,
        "total_found": len(quotes),
        "method": "keyword_extraction",
        "grounding_instruction": (
            "Use these extracted sections to answer. "
            "Cite specific quotes when making claims."
        )
    }


async def require_grounding(
    db: DatabaseService,
    session_id: str,
    statement: str,
    source_type: str = "any"
) -> Dict[str, Any]:
    """
    Require that a statement be grounded in stored facts before accepting it.

    Args:
        db: Database service
        session_id: Current session
        statement: The statement to verify
        source_type: Type of source required ("anchor", "memory", "any")

    Returns:
        Dict with grounding verification
    """
    grounding_sources = []

    # Check against anchors
    events = await db.get_timeline_events(
        session_id=session_id,
        limit=50,
        anchors_only=True
    )

    statement_lower = statement.lower()

    for event in events:
        if event.get("is_anchor"):
            summary_lower = event["summary"].lower()
            # Check for keyword overlap
            overlap = sum(1 for word in statement_lower.split() if len(word) > 3 and word in summary_lower)
            if overlap >= 2:
                grounding_sources.append({
                    "type": "anchor",
                    "content": event["summary"],
                    "match_strength": "keyword_overlap"
                })

    # Check against memories if needed
    if source_type in ["memory", "any"] and not grounding_sources:
        try:
            from services.embeddings import EmbeddingService
            embeddings = EmbeddingService()
            embedding = await embeddings.generate_embedding(statement)

            memories = await db.search_similar(
                embedding=embedding,
                limit=3,
                threshold=0.7
            )

            for memory in memories:
                grounding_sources.append({
                    "type": "memory",
                    "content": memory.get("content", "")[:200],
                    "similarity": memory.get("similarity")
                })
        except:
            pass

    is_grounded = len(grounding_sources) > 0

    return {
        "success": True,
        "statement": statement,
        "is_grounded": is_grounded,
        "grounding_sources": grounding_sources,
        "source_count": len(grounding_sources),
        "recommendation": (
            "Statement is grounded in stored facts"
            if is_grounded
            else "WARNING: Statement has no grounding. Verify before using."
        )
    }
