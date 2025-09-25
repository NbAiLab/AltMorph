# AltMorph Code Walkthrough

This Norwegian morphological analysis tool finds alternative word forms in context. It can, for instance, show that "kasta" might correspond to "kastet" in some contexts, but not others.

## The Big Picture: What Does This System Do?

Imagine you have the sentence "Katta ligger på matten." AltMorph will analyze each word and produce:
`"{Katta, Katten} ligger på {matten, matta}."`

This means:
- "Katta" could also be "Katten"  
- "matten" could also be "matta"
- "ligger" and "på" remain unchanged as they have no relevant alternatives in this context

The important detail is that **context matters**. The same word "matta" might have different alternatives depending on where it appears in the sentence and what grammatical role it plays. It could be "Han matta kongen i sjakk". Here, the word "matten" is not an alternative.

## Architecture Overview: The Processing Pipeline

The processing pipeline runs through these main stages:

```
Input Sentence → Tokenization → POS Tagging → API Lookup → BERT Filtering → Output
```

Let's trace through each stage with examples.

## Stage 1: Text Preprocessing and Tokenization

### The Punctuation Problem

First challenge: How do you handle `"matten."` vs `"matten ."`? 

```python
def preprocess_punctuation(text: str) -> str:
    """Insert spaces before punctuation to ensure proper tokenization."""
    return re.sub(r'(?<!\s)([,.;:?!])', r' \1', text)
```

**Why this matters**: The Norwegian Ordbank API can't find "matta." but it CAN find "matta". So we:
1. **Preprocess**: `"Katta ligger på matta."` → `"Katta ligger på matta ."`
2. **Process**: Find alternatives for clean words
3. **Postprocess**: `"Katta ligger på {matta, matten} ."` → `"Katta ligger på {matta, matten}."`

### Tokenization

```python
def tokenize_preserve(text: str) -> List[str]:
    """Tokenize text while preserving exact spacing and punctuation."""
    return re.findall(r'\S+|\s+', text)
```

This gives us: `["Katta", " ", "ligger", " ", "på", " ", "matta", " ", "."]`

**Why preserve spaces?** Because when we reconstruct the sentence later, we want the exact original formatting. No weird extra spaces or missing punctuation.

## Stage 2: Part-of-Speech (POS) Tagging

### The Norwegian BERT POS Tagger

```python
@lru_cache(maxsize=1)
def get_pos_tagger():
    """Load POS tagger model (lazy initialization)."""
    tagger = pipeline(
        "token-classification",
        model="NbAiLab/nb-bert-base-pos",
        aggregation_strategy="none"
    )
    return tagger
```

**Why POS tagging?** Because "ligger" could be:
- A **noun** (the position/location)
- A **verb** (to lie/be positioned)

The API gives different results for nouns vs verbs, so we need to know what we're looking for.

### Handling Sub-word Tokenization

BERT tokenizes "kastene" into ["kast", "##ene"]. We need to reconstruct this:

```python
def extract_pos_tags(sentence: str) -> Dict[str, str]:
    pos_results = tagger(sentence)
    
    word_pos_map = {}
    current_word = ""
    current_pos = None
    
    for token_info in pos_results:
        token = token_info['word']
        pos_tag = token_info['entity']
        
        if token.startswith('##'):
            current_word += token[2:]  # Remove ## prefix and append
        else:
            # Save previous word if it exists
            if current_word and current_pos:
                word_pos_map[current_word.lower()] = current_pos
            # Start new word
            current_word = token
            current_pos = pos_tag
```

**Example output**:
```
katta: NOUN
ligger: VERB
på: ADP
matta: NOUN
```

## Stage 3: API Lookup Strategy

This stage gathers every form of a word that might apply.

### The Multi-Lemma Challenge

Here's a real example: "matten" (the mat) has multiple forms across different lemmas in the Norwegian dictionary:

1. **Lemma 43856**: matte (NOUN) → matte, matten, matter, mattene
2. **Lemma 43857**: matte (NOUN) → matte, matta, matter, mattene

**The problem**: If we only looked at the first lemma, we'd miss "matta" as an alternative. But if we combine all lemmas blindly, we get irrelevant forms.

### Lemma Selection

```python
def get_alternatives(word: str, lang: str, headers: Dict[str, str], timeout: float,
                    pos_filter: Optional[str] = None, debug: bool = False) -> Optional[Set[str]]:
    """Get alternative forms for a word."""
    
    # Step 1: Search for ALL lemmas that might contain this word
    lemmas = search_lemmas(word, lang, headers, timeout, pos_filter, debug)
    
    # Step 2: For each lemma, get its inflections and check if it contains our word
    matching_lemmas = []
    target_word_lower = word.casefold()
    
    for lemma in lemmas:
        lemma_id = int(lemma["id"])
        inflections = collect_inflections([lemma_id], lang, headers, timeout, debug)
        
        # Does this lemma actually contain our target word?
        contains_word = any(inf["word_form"].casefold() == target_word_lower 
                           for inf in inflections)
        
        if contains_word:
            matching_lemmas.append({"id": lemma_id, "inflections": inflections})
```

**Why this approach?** We only use lemmas that actually contain the word we're analyzing. This prevents "matten" from picking up forms from unrelated lemmas.

### API Caching: Performance Optimization

```python
def search_lemmas(word: str, lang: str, headers: Dict[str, str], timeout: float,
                 pos_filter: Optional[str] = None, debug: bool = False) -> List[Dict]:
    
    # Check cache first
    cache_key = make_cache_key("lemmas", word.casefold(), lang, pos_filter or "None")
    cached_result = load_from_cache(cache_key)
    if cached_result is not None:
        if debug:
            logger.debug("💾 CACHE HIT: lemmas for '%s'", word)
        return cached_result
    
    # ... API call logic ...
    
    # Save to cache before returning
    save_to_cache(cache_key, result)
    return result
```

**Why caching?** API calls are slow (200-500ms each). Without caching:
- First run of "Katta ligger på matta": ~2-3 seconds
- With caching: subsequent runs are ~100ms

**Cache key strategy**: `lemmas_katta_nob_NOUN` ensures we cache different results for the same word with different POS tags.

### JSON Serialization Handling

When caching data structures, we need to handle the conversion between Python types and JSON:

```python
def load_from_cache(cache_key: str) -> Optional[any]:
    # ... load from JSON ...
    
    # Convert tags from lists back to tuples after JSON loading
    if isinstance(data, list) and data and isinstance(data[0], dict) and "tags" in data[0]:
        for item in data:
            if "tags" in item and isinstance(item["tags"], list):
                item["tags"] = tuple(item["tags"])  # Convert back to tuple
```

**Why this conversion?** Python tuples become JSON arrays during serialization, but our code expects tuples for use as dictionary keys and set elements.

## Stage 4: Grammatical Tag Matching

Once we have all inflections, we need to find which grammatical "slot" our word fills:

```python
def find_matching_tags(target_word: str, inflections: List[Dict], debug: bool = False) -> Set[Tuple[str, ...]]:
    target_lower = target_word.casefold()
    matching_tags = set()
    
    for inflection in inflections:
        if inflection["word_form"].casefold() == target_lower:
            matching_tags.add(inflection["tags"])
```

**Example for "matta"**:
- Found inflection: `word_form='matta', tags=('Sing', 'Ind')`
- This means "matta" is singular indefinite
- So we only want alternatives that are also singular indefinite

**Example output**:
```
🏷️ FINDING TAGS FOR: matta
   Found match: matta -> ('Sing', 'Ind')
   Final matching tags: {('Sing', 'Ind')}

🔍 COLLECTING ALTERNATIVES WITH MATCHING TAGS:
   ✅ matta (tags: ('Sing', 'Ind'))  
   ✅ matten (tags: ('Sing', 'Def'))   # Definite form alternative!
   ❌ matter (tags: ('Plur', 'Ind')) - doesn't match  # Wrong number
```

## Stage 5: BERT-Based Acceptability Filtering

Not all grammatically correct alternatives make sense in context, so we score them before presenting the output.

### The Context Problem

Consider: `"Katta ligger på den matta som er der"`

- "matta" could theoretically be "matter" (grammatically valid)
- But "matter" (plural) makes no sense with "den" (singular)

### Position-Specific Analysis

```python
def score_word_in_context(sentence: str, target_word: str, target_position: Optional[int] = None) -> Dict:
    """Score a word's acceptability in its sentence context."""
    
    # Create masked version of sentence at specific position
    words = sentence.split()
    masked_words = words.copy()
    masked_words[target_idx] = tokenizer.mask_token  # Replace with [MASK]
    masked_sentence = " ".join(masked_words)
    
    # Ask BERT: how likely is this word in this position?
    inputs = tokenizer(masked_sentence, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_pos]
        probabilities = torch.softmax(logits, dim=0)
```

**Example**:
```
Original: "Katta ligger på den matta som er der"
Masked:   "Katta ligger på den [MASK] som er der"

BERT scores:
- matta: logit=7.521 (very likely)
- matter: logit=1.497 (unlikely - doesn't fit with "den")
```

### The Logit Threshold Strategy

```python
def filter_by_acceptability(tokens: List[str], position: int, alternatives: Set[str],
                          threshold: float = 2.0, debug: bool = False) -> Set[str]:
    
    original_score = score_word_in_context(original_sentence, original_word, position)
    
    for alt in alternatives:
        score = score_word_in_context(test_sentence, alt, position)
        logit_diff = original_score['logit'] - score['logit']
        
        if logit_diff <= threshold:
            filtered.add(alt)  # Keep it
        else:
            # Reject - too unlikely compared to original
```

**Why logits instead of probabilities?** Probabilities are tiny (0.000001) and hard to work with. Logits are the raw scores before softmax, and differences in logits correspond to ratios in probabilities:

- Logit difference of 2.0 ≈ 7x less likely
- Logit difference of 3.0 ≈ 20x less likely

**Example filtering**:
```
🔍 ANALYZING: matta (position 3)
Context: Katta ligger på den [matta] som er der.

matta    : Logit 7.521 (ORIGINAL)
matter   : Logit 1.497
❌ REJECTING matter: logit diff +6.023 > 2.0 (too improbable)
```

### Position Matters

The same word can have different alternatives in different positions:

```python
# Position 3: "Katta ligger på [matta]"  
matta: acceptable alternatives = {matta, matten}

# Position 5: "ligger på den [matta] som"
matta: acceptable alternatives = {matta} only
```

**Why?** Because the grammatical context is different. In position 3, both indefinite and definite forms work. In position 5 with "den", only the indefinite form fits grammatically.

## Stage 6: Output Construction and Case Matching

### Preserving Original Formatting

```python
def case_match(original: str, target: str) -> str:
    """Match the casing pattern of original string to target."""
    if original.islower():
        return target.lower()
    elif original.isupper():
        return target.upper()
    elif original.istitle():
        return target.capitalize()
    return target
```

**Why this matters**: If the input is "Katta", the output should be "{Katta, Katten}", not "{katta, katten}".

### Ordering

```python
# Order: original first, then others sorted
original = case_match(token, normalized[token.casefold()])
others = sorted([
    case_match(token, alt) for key, alt in normalized.items()
    if key != token.casefold()
], key=str.casefold)

ordered = [original] + others
output_parts.append("{" + ", ".join(ordered) + "}")
```

This ensures consistent output: `{original, alternative1, alternative2}` rather than random ordering.

## Performance Optimizations

### Concurrent API Calls

```python
with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {}
    
    for word in unique_words:
        future = executor.submit(get_alternatives, word, lang, headers, timeout)
        futures[future] = word
    
    for future in cf.as_completed(futures):
        word = futures[future]
        result = future.result()
```

**Why threading?** API calls are I/O bound. Instead of:
- Sequential: word1 (500ms) → word2 (500ms) → word3 (500ms) = 1.5 seconds
- Concurrent: [word1, word2, word3] in parallel = 500ms total

### Lazy Model Loading

```python
@lru_cache(maxsize=1)
def get_pos_tagger():
    """Load POS tagger model (lazy initialization)."""
    # Model only loaded when first needed, then cached
```

**Why lazy loading?** BERT models are large (400MB+). Only load them when actually needed, and never load them twice.

## Error Handling and Resilience

### HTTP Retry Logic

```python
def http_get(url: str, headers: Dict[str, str], timeout: float) -> Optional[List]:
    """HTTP GET with retries."""
    for attempt in range(3):
        try:
            response = SESSION.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(0.75)  # Wait before retry
    return None
```

**Why retries?** Network calls fail. Better to retry than to crash the entire analysis.

### Graceful Degradation

```python
try:
    result = future.result()
    cache[word] = result
except Exception as e:
    cache[word] = None  # Mark as failed, but continue processing
```

If one word fails, we continue with the others rather than crashing the entire sentence.

## Debugging and Verbosity

The system has 4 verbosity levels:

- **Level 0**: Silent (just the result)
- **Level 1**: Basic progress (`Loading models...`)
- **Level 2**: Processing details (what words found, API results)
- **Level 3**: Full debug (BERT scores, filtering decisions, cache hits)

**Example Level 3 output**:
```
🎯 PROCESSING: Katta ligger på matta.

📝 WORDS: ['katta', 'ligger', 'på', 'matta']

🏷️ POS TAGS:
   katta: NOUN
   ligger: VERB
   på: ADP
   matta: NOUN

📡 API LOOKUP: katta (POS: NOUN)
💾 CACHE HIT: lemmas for 'katta' (POS: NOUN)
   ✅ katta: 2 alternatives: ['katta', 'katten']

🧠 ACCEPTABILITY FILTERING (threshold: 3.00)

🔍 ANALYZING: katta (position 0)
   Context: [Katta] ligger på matta.
   Alternatives: ['katta', 'katten']
     katten      : Logit 3.017, Prob 7.56e-05, Rank -1
     katta       : Logit 3.155, Prob 2.83e-05, Rank -1 (ORIGINAL)
     ✅ KEEPING katten: logit diff +0.138 <= 3.0
     📊 Result: kept 2, rejected 0
```

## Key Design Decisions and Rationale

### Why BERT for Acceptability?
- **Alternative**: Use grammar rules or simple statistics
- **Why BERT**: Captures subtle contextual relationships that rules can't express
- **Trade-off**: Slower but much more accurate

### Why Position-Specific Analysis?
- **Problem**: Same word, different contexts need different treatment
- **Solution**: Analyze each word occurrence separately
- **Cost**: More BERT calls, but essential for accuracy

### Why Comprehensive Lemma Search?
- **Problem**: Missing valid alternatives if we only check first lemma
- **Solution**: Check all lemmas, but only use those containing target word
- **Trade-off**: More API calls, but complete coverage

### Why Caching?
- **Problem**: API calls are slow (300-500ms each)
- **Solution**: File-based cache with explicit key generation
- **Result**: 10x speed improvement on subsequent runs

## Summary

AltMorph combines:
1. **Norwegian POS tagging** for linguistic accuracy
2. **Comprehensive API querying** for completeness  
3. **BERT contextual scoring** for acceptability
4. **Caching and concurrency** for performance
5. **Robust error handling** for reliability

The key insight is that morphological alternatives aren't just about grammar—context matters enormously. A word that's grammatically correct might be contextually inappropriate, and only advanced language models can make these subtle distinctions.

The system prioritizes correctness over speed, but uses caching and concurrency to make it practical for real-world use. Every design decision aims to handle the complexity of natural language while providing reliable results.

When you see the output `"{Katta, Katten} ligger på {matten, matta}."`, the pipeline has considered grammar, context, and acceptability to keep alternatives that are likely to fit.
