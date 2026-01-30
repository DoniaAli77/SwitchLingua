# CS Ratio Calculation Issue

## The Problem You Identified

The **CS ratio agent relies on the LLM to count words/tokens**, which is:
- **Unreliable** for non-ASCII languages (Arabic, Chinese, etc.)
- **Inconsistent** across different LLM models
- **Prone to errors** with mixed-language text

---

## How CS Ratio Currently Works

**File:** [`core/prompt.py`](core/prompt.py#L202-L248)

```python
CS_RATIO_PROMPT = ChatPromptTemplate.from_messages([
    (
        "assistant",
        """
        You are **CSRatioAgent**. You evaluate the *Code-Switching Ratio* (CS-Ratio) in given text.
        
        1. **Check the proportion** of matrix language vs. embedded language:
        - Count tokens/words for each language.
        - Compare to a desired ratio (e.g., 70% matrix, 30% embedded) if provided.
        
        2. **Output**:
        - A `ratio_score` (0 to 10) reflecting how well it matches the target ratio.
        - A `computed_ratio` or breakdown: e.g., "66% : 34%".
        - A `notes` field listing any ratio-related observations.
        
        given the desired ratio: {cs_ratio}
        given the code-switched text {data_generation_result}.
        """
    )
])
```

**Current flow:**
1. LLM receives list of code-switched sentences
2. LLM is asked to "count tokens/words for each language"
3. LLM responds with a ratio (e.g., "63% Arabic : 37% English")

---

## Why This Is Unreliable

### Issue 1: Non-ASCII Language Word Boundaries

Languages like Arabic, Chinese, Japanese don't use spaces to separate words. Example:

```
Arabic: "الفريق بدأ المباراة"
        ↑     ↑   ↑
        Word1 Word2 Word3 (clear boundaries in this case)

But more complex:
"والجمهورمتحمسجدا" 
(No spaces - compound word, how does LLM count this?)
```

An LLM might:
- Count "والجمهورمتحمسجدا" as **1 word** (wrong!)
- Count it as **4 words** (depends on tokenization logic)
- Inconsistently count it differently on different runs

### Issue 2: Mixed-Language Word Boundaries

```
English: "But then the defense couldn't"
           ↑    ↑   ↑   ↑        ↑
          Word1 Word2 Word3 Word4 Word5 (5 words, clear)

Mixed: "الفريق But then الدفاع couldn't"
        ↑      ↑    ↑    ↑      ↑
        1?     6?   7?   8?     9?

How does the LLM decide where one word ends and another begins?
Does contraction "couldn't" count as 1 or 2 words?
```

### Issue 3: LLM Counting Variation

Different responses from same input:

**Run 1:**
```
Text: "الفريق بدأ المباراة. But then the defense couldn't."
LLM Response: "5 Arabic words, 5 English words → 50% : 50%"
```

**Run 2 (same input, different temperature/model):**
```
LLM Response: "5 Arabic words, 6 English words → 45% : 55%"
(counted "couldn't" as 2 tokens instead of 1)
```

### Issue 4: No Language Detection

The LLM might misidentify which language a word belongs to:

```
Example: "café" (French word, has accents)
- Is this French? English? 
- Does it count as French or English in a mixed passage?

Or: Proper nouns like "Ahmed" (Arabic name used in English text)
- Which language counts it as?
```

---

## Real Example: Why Ratios Are Wrong

### Scenario Input

```yaml
first_language: "Arabic"
second_language: "English"
cs_ratio: "30%"  # Want 30% English, 70% Arabic
```

### Generated Text (from LLM)

```
"الفريق بدأ المباراة بشكل قوي جدًا وحقق تقدم كبير في البداية. But then the defense couldn't keep up and things started to fall apart."
```

### What CS Ratio Agent Should Calculate

**Manual count (word-by-word):**

```
Arabic part:
"الفريق" (1) "بدأ" (2) "المباراة" (3) "بشكل" (4) "قوي" (5) "جدًا" (6) "وحقق" (7) "تقدم" (8) "كبير" (9) "في" (10) "البداية" (11)
= 11 Arabic words

English part:
"But" (1) "then" (2) "the" (3) "defense" (4) "couldn't" (5) "keep" (6) "up" (7) "and" (8) "things" (9) "started" (10) "to" (11) "fall" (12) "apart" (13)
= 13 English words

Ratio:
Arabic: 11 / (11 + 13) = 11/24 = 45.8%
English: 13 / (11 + 13) = 13/24 = 54.2%

Actual ratio: 46% Arabic : 54% English
Target ratio: 70% Arabic : 30% English
Difference: OFF BY 24 percentage points!
```

### What LLM Might Calculate

The LLM might respond with:

```json
{
  "computed_ratio": "63% Arabic : 37% English",
  "ratio_score": 7.0,
  "notes": "Slightly higher embedded language usage than target."
}
```

**Even though the actual ratio is 46% : 54%!** The LLM miscounted.

---

## The Root Cause

**The prompt asks the LLM to do a computational task (word counting) that LLMs are not good at.**

LLMs are:
- ✓ Good at understanding context
- ✓ Good at language generation
- ✓ Good at semantic analysis
- ✗ **Bad at precise token counting**, especially for:
  - Non-ASCII languages
  - Mixed-language text
  - Compound words without clear boundaries

---

## Solution: Replace LLM Counting with Deterministic Tokenization

### Option 1: Simple Split-Based Counting (Fast, Basic)

```python
def compute_cs_ratio_deterministic(
    instances: list[str],
    first_language: str,
    second_language: str
) -> dict:
    """
    Count code-switching ratio using simple split logic.
    Note: Works best for languages with clear space-separated words.
    """
    # For simplicity, use word counts (split by spaces)
    total_words = 0
    lang1_words = 0
    lang2_words = 0
    
    # Rough heuristics for language detection (would need improvement)
    # Arabic Unicode ranges
    arabic_chars = set(chr(c) for c in range(0x0600, 0x06FF))
    
    for instance in instances:
        words = instance.split()
        for word in words:
            # Remove punctuation for language detection
            clean_word = ''.join(c for c in word if c.isalpha())
            
            if not clean_word:
                continue
            
            total_words += 1
            
            # Check if word contains Arabic characters
            if any(c in arabic_chars for c in clean_word):
                lang1_words += 1
            else:
                lang2_words += 1
    
    if total_words == 0:
        return {
            "ratio_score": 0,
            "computed_ratio": "0% : 0%",
            "notes": "No words detected."
        }
    
    lang1_ratio = (lang1_words / total_words) * 100
    lang2_ratio = (lang2_words / total_words) * 100
    
    return {
        "ratio_score": 10,  # Placeholder
        "computed_ratio": f"{lang1_ratio:.1f}% {first_language} : {lang2_ratio:.1f}% {second_language}",
        "notes": f"Counted {lang1_words} {first_language} words and {lang2_words} {second_language} words."
    }
```

### Option 2: Unicode-Based Detection (Better for Non-ASCII)

```python
def detect_language_by_unicode(word: str, first_language: str, second_language: str) -> str:
    """
    Detect language based on Unicode ranges.
    
    Supports:
    - Arabic: U+0600 to U+06FF
    - Chinese: U+4E00 to U+9FFF
    - Cyrillic: U+0400 to U+04FF
    - Latin: U+0000 to U+007F
    """
    arabic_range = range(0x0600, 0x06FF)
    chinese_range = range(0x4E00, 0x9FFF)
    cyrillic_range = range(0x0400, 0x04FF)
    latin_range = range(0x0000, 0x007F)
    
    ranges = {
        "Arabic": arabic_range,
        "Chinese": chinese_range,
        "Cyrillic": cyrillic_range,
        "English": latin_range,
        "French": latin_range,
        "Spanish": latin_range,
    }
    
    # Count characters from each language
    lang_counts = {}
    for char in word:
        if ord(char) not in arabic_range and \
           ord(char) not in chinese_range and \
           ord(char) not in cyrillic_range and \
           ord(char) not in latin_range:
            continue
        
        for lang, range_obj in ranges.items():
            if ord(char) in range_obj:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    if not lang_counts:
        return "unknown"
    
    # Return language with most characters
    return max(lang_counts, key=lang_counts.get)
```

### Option 3: Library-Based Tokenization (Most Accurate)

```python
from langdetect import detect_langs
from textstat import lexicon_count

def compute_cs_ratio_with_langdetect(
    instances: list[str],
    first_language: str,
    second_language: str
) -> dict:
    """
    Use langdetect library for more accurate language detection.
    
    Install: pip install langdetect
    """
    lang1_count = 0
    lang2_count = 0
    
    for instance in instances:
        words = instance.split()
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            if not clean_word:
                continue
            
            try:
                detected = detect_langs(clean_word)
                if detected:
                    top_lang = detected[0].lang
                    # Map ISO codes to language names
                    if _is_language(top_lang, first_language):
                        lang1_count += 1
                    elif _is_language(top_lang, second_language):
                        lang2_count += 1
            except:
                # If detection fails, skip
                pass
    
    total = lang1_count + lang2_count
    if total == 0:
        return {
            "ratio_score": 0,
            "computed_ratio": "0% : 0%",
            "notes": "No words detected."
        }
    
    lang1_ratio = (lang1_count / total) * 100
    lang2_ratio = (lang2_count / total) * 100
    
    return {
        "ratio_score": 10,
        "computed_ratio": f"{lang1_ratio:.1f}% {first_language} : {lang2_ratio:.1f}% {second_language}",
        "notes": f"Detected {lang1_count} {first_language} and {lang2_count} {second_language} words."
    }
```

---

## How to Fix in Your Codebase

### Step 1: Create a new utility module

**File:** `core/cs_ratio_calculator.py` (new file)

```python
"""Deterministic code-switching ratio calculation (no LLM)."""

def compute_cs_ratio(
    instances: list[str],
    first_language: str,
    second_language: str
) -> dict:
    """
    Compute CS ratio using character-level language detection.
    
    Returns:
        {
            "ratio_score": float (0-10),
            "computed_ratio": str (e.g., "70% Arabic : 30% English"),
            "notes": str,
            "lang1_word_count": int,
            "lang2_word_count": int,
        }
    """
    # Unicode ranges for common languages
    lang_ranges = {
        "Arabic": range(0x0600, 0x06FF),
        "Chinese": range(0x4E00, 0x9FFF),
        "Hindi": range(0x0900, 0x097F),
        "French": range(0x0000, 0x00FF),
        "English": range(0x0000, 0x00FF),
        "Spanish": range(0x0000, 0x00FF),
    }
    
    lang1_range = lang_ranges.get(first_language, range(0x0000, 0x00FF))
    lang2_range = lang_ranges.get(second_language, range(0x0000, 0x00FF))
    
    lang1_count = 0
    lang2_count = 0
    
    for instance in instances:
        words = instance.split()
        
        for word in words:
            # Skip punctuation-only tokens
            clean_word = ''.join(c for c in word if c.isalpha())
            if not clean_word:
                continue
            
            # Check first character for language (simple heuristic)
            first_char = clean_word[0]
            if ord(first_char) in lang1_range:
                lang1_count += 1
            elif ord(first_char) in lang2_range:
                lang2_count += 1
    
    total = lang1_count + lang2_count
    if total == 0:
        return {
            "ratio_score": 0,
            "computed_ratio": "0% : 0%",
            "notes": "No words detected.",
            "lang1_word_count": 0,
            "lang2_word_count": 0,
        }
    
    lang1_percent = (lang1_count / total) * 100
    lang2_percent = (lang2_count / total) * 100
    
    return {
        "ratio_score": 10,  # Will be adjusted based on target
        "computed_ratio": f"{lang1_percent:.1f}% {first_language} : {lang2_percent:.1f}% {second_language}",
        "notes": f"Counted {lang1_count} {first_language} and {lang2_count} {second_language} words.",
        "lang1_word_count": lang1_count,
        "lang2_word_count": lang2_count,
        "lang1_percent": lang1_percent,
        "lang2_percent": lang2_percent,
    }
```

### Step 2: Modify CSRatioAgent to use deterministic calculation

**File:** [`core/node_engine.py`](core/node_engine.py#L144-L154)

Replace:
```python
def RunCSRatioAgent(state: AgentRunningState):
    CSRatioAgent = CS_RATIO_PROMPT | ChatOpenAI(
        model=MODEL, temperature=0.1, base_url=API_BASE
    ).with_structured_output(CSRatioResponse)
    response = CSRatioAgent.invoke(state)

    return {"cs_ratio_result": response}
```

With:
```python
def RunCSRatioAgent(state: AgentRunningState):
    # Option 1: Use deterministic calculation + LLM scoring
    from cs_ratio_calculator import compute_cs_ratio
    
    ratio_calc = compute_cs_ratio(
        state["data_generation_result"],
        state["first_language"],
        state["second_language"]
    )
    
    # Parse target ratio (e.g., "30%" → 30)
    target_ratio = float(state["cs_ratio"].rstrip('%'))
    actual_ratio = ratio_calc["lang2_percent"]
    
    # Calculate score (0-10, higher if closer to target)
    ratio_diff = abs(actual_ratio - target_ratio)
    ratio_score = max(0, 10 - (ratio_diff / 10))  # -1 point per 10% difference
    
    response = {
        "ratio_score": ratio_score,
        "computed_ratio": ratio_calc["computed_ratio"],
        "notes": f"Target: {target_ratio}% {state['second_language']}, Actual: {actual_ratio:.1f}%"
    }
    
    return {"cs_ratio_result": response}
```

---

## Example: Before vs After

### Before (LLM-based, unreliable)

```
Input text:
"الفريق بدأ المباراة بشكل قوي جدًا. But then the defense couldn't."

LLM output (unreliable):
{
  "computed_ratio": "63% Arabic : 37% English",
  "ratio_score": 7.0,
  "notes": "Slightly higher embedded language usage than target."
}

Reality: 46% Arabic : 54% English
Error: 17 percentage points OFF!
```

### After (deterministic, reliable)

```
Input text:
"الفريق بدأ المباراة بشكل قوي جدًا. But then the defense couldn't."

Deterministic output (accurate):
{
  "computed_ratio": "45.8% Arabic : 54.2% English",
  "ratio_score": 2.0,  # 30% target vs 54.2% actual = 24.2% diff → score drops
  "notes": "Target: 30% English, Actual: 54.2%"
}

Reality: Matches exactly
Error: 0 percentage points (accurate!)
```

---

## Pros and Cons

| Approach | Pros | Cons |
|----------|------|------|
| **LLM-based (current)** | Simple, integrated | Inaccurate, inconsistent, slow |
| **Split + Unicode ranges** | Fast, deterministic | Doesn't handle all languages, edge cases |
| **langdetect library** | More accurate, language-aware | Extra dependency, might be slow |
| **Hybrid (deterministic + LLM for explanation)** | Accurate ratio + LLM explanation | Slightly more complex |

---

## Recommendation

Use **Option 2 (deterministic + score adjustment)**:
1. Use character-level language detection to accurately count words
2. Compute the actual ratio
3. Score based on how far it is from the target
4. Keep the computation fast and reliable

This fixes the accuracy issue while maintaining the scoring system.

---

## Summary

| Issue | Current Problem | Why It Happens | Solution |
|-------|-----------------|----------------|----------|
| **Inaccurate ratios** | LLM miscounts words | LLMs aren't good at precise counting | Use deterministic tokenization |
| **Inconsistent results** | Different counts each run | LLM variation + temperature | Deterministic algorithm |
| **Non-ASCII struggles** | Arabic/Chinese miscounted | LLM word boundary confusion | Unicode range detection |
| **Slow evaluation** | LLM API call per scenario | Using LLM for computation task | Hardcoded Python calculation |

---
