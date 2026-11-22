def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|; 0 for two empty sets."""
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0
