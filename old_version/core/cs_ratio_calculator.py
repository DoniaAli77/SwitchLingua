"""
Deterministic Code-Switching Ratio Calculator

Uses Unicode character ranges to accurately detect languages and compute
word counts, eliminating LLM inconsistencies in ratio calculation.
"""

from typing import Dict, Any, Tuple


# Unicode ranges for common languages
LANGUAGE_RANGES = {
    "Arabic": (0x0600, 0x06FF),
    "Chinese": (0x4E00, 0x9FFF),
    "Hindi": (0x0900, 0x097F),
    "French": (0x0000, 0x00FF),
    "English": (0x0000, 0x00FF),
    "Spanish": (0x0000, 0x00FF),
    "German": (0x0000, 0x00FF),
    "Japanese": (0x3040, 0x309F),  # Hiragana
    "Vietnamese": (0x0000, 0x00FF),
    "Bengali": (0x0980, 0x09FF),
    "Thai": (0x0E00, 0x0E7F),
    "Korean": (0xAC00, 0xD7AF),
    "Russian": (0x0400, 0x04FF),
}


def get_language_range(language: str) -> Tuple[int, int]:
    """
    Get the Unicode range for a language.
    
    Args:
        language: Language name (e.g., "Arabic", "English")
    
    Returns:
        Tuple of (start, end) Unicode code points
        Defaults to Latin range (0x0000, 0x00FF) if language not found
    """
    return LANGUAGE_RANGES.get(language, (0x0000, 0x00FF))


def detect_word_language(word: str, lang1: str, lang2: str) -> str:
    """
    Detect which language a word belongs to based on its characters.
    
    Strategy:
    1. Skip punctuation-only tokens
    2. Check the first alphabetic character's Unicode range
    3. Return the matching language, or "unknown" if no match
    
    Args:
        word: The word to classify
        lang1: First language name
        lang2: Second language name
    
    Returns:
        "lang1", "lang2", or "unknown"
    """
    # Extract only alphabetic characters
    clean_word = "".join(c for c in word if c.isalpha())
    
    if not clean_word:
        return "unknown"
    
    # Get language ranges
    lang1_start, lang1_end = get_language_range(lang1)
    lang2_start, lang2_end = get_language_range(lang2)
    
    # Check first character
    first_char_code = ord(clean_word[0])
    
    if lang1_start <= first_char_code <= lang1_end:
        return "lang1"
    elif lang2_start <= first_char_code <= lang2_end:
        return "lang2"
    else:
        return "unknown"


def compute_cs_ratio(
    instances: list,
    first_language: str,
    second_language: str
) -> Dict[str, Any]:
    """
    Compute code-switching ratio using deterministic tokenization.
    
    This avoids LLM inconsistencies by:
    - Using character-based language detection (Unicode ranges)
    - Splitting on whitespace (works for most languages)
    - Counting words accurately
    
    Args:
        instances: List of generated code-switched sentences
        first_language: Matrix language (dominant)
        second_language: Embedded language (secondary)
    
    Returns:
        Dict with:
        {
            "lang1_word_count": int,
            "lang2_word_count": int,
            "lang1_percent": float,
            "lang2_percent": float,
            "computed_ratio": str,  # e.g., "70% Arabic : 30% English"
        }
    """
    lang1_count = 0
    lang2_count = 0
    unknown_count = 0
    
    for instance in instances:
        # Split by whitespace to get words
        words = instance.split()
        
        for word in words:
            detected = detect_word_language(word, first_language, second_language)
            
            if detected == "lang1":
                lang1_count += 1
            elif detected == "lang2":
                lang2_count += 1
            else:
                unknown_count += 1
    
    total = lang1_count + lang2_count
    
    if total == 0:
        return {
            "lang1_word_count": 0,
            "lang2_word_count": 0,
            "lang1_percent": 0.0,
            "lang2_percent": 0.0,
            "computed_ratio": "0% : 0%",
            "details": "No words detected in text.",
        }
    
    lang1_percent = (lang1_count / total) * 100
    lang2_percent = (lang2_count / total) * 100
    
    # NOTE: `cs_ratio` semantics in this project refer to the *second_language*
    # (embedded language) percentage. To avoid confusion, format the
    # `computed_ratio` with the embedded (second) language first, e.g.
    # "30.0% English : 70.0% Arabic" when English is the embedded language.
    return {
        "lang1_word_count": lang1_count,
        "lang2_word_count": lang2_count,
        "lang1_percent": lang1_percent,
        "lang2_percent": lang2_percent,
        "computed_ratio": f"{lang2_percent:.1f}% {second_language} : {lang1_percent:.1f}% {first_language}",
        "unknown_words": unknown_count,
        "total_words": total,
        "details": f"Counted {lang1_count} {first_language} words and {lang2_count} {second_language} words.",
    }


def calculate_ratio_score(
    actual_percent: float,
    target_ratio_str: str,
    max_points: int = 10
) -> float:
    """
    Calculate a score (0-10) based on how close actual ratio is to target.
    
    Scoring:
    - Perfect match (0% diff): 10 points
    - 1-5% diff: 9-8 points
    - 6-10% diff: 7-5 points
    - 11-20% diff: 4-2 points
    - >20% diff: 0-1 points
    
    Args:
        actual_percent: Actual percentage of embedded language (0-100)
        target_ratio_str: Target ratio as string, e.g., "30%"
        max_points: Maximum score (default 10)
    
    Returns:
        Score from 0 to max_points
    """
    try:
        target_percent = float(target_ratio_str.rstrip("%"))
    except (ValueError, AttributeError):
        # If target can't be parsed, return neutral score
        return max_points / 2
    
    diff = abs(actual_percent - target_percent)
    
    # Linear penalty: lose 0.5 points per 1% difference
    score = max(0, max_points - (diff * 0.5))
    
    return score
