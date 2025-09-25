# AltMorph Code Walkthrough: Senior to Junior Developer

Welcome! Let's walk through the AltMorph codebase together. This is a sophisticated Norwegian morphological analysis tool that finds alternative word forms in context. Think of it as a smart system that can tell you that "kasta" could also be "kastet" in some contexts, but not others.

## The Big Picture: What Does This System Do?

Imagine you have the sentence "Jenta kasta ballen til gutten." AltMorph will analyze each word and produce:
`"{Jenta, Jenten} {kasta, kastet} {ballen, balla} til gutten."`

This means:
- "Jenta" could also be "Jenten" (subject vs. object form)  
- "kasta" could also be "kastet" (past tense variants)
- "ballen" could also be "balla" (dialectal variant)

But here's the clever part: **context matters**. The same word "kasta" might have different alternatives depending on where it appears in the sentence and what grammatical role it plays.

## Architecture Overview: The Processing Pipeline

The system works like a assembly line with these main stages:

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

**Why this matters**: The Norwegian Ordbank API can't find "matten." but it CAN find "matten". So we:
1. **Preprocess**: `"Katta ligger på matten."` → `"Katta ligger på matten ."`
2. **Process**: Find alternatives for clean words
3. **Postprocess**: `"Katta ligger på {matten, matta} ."` → `"Katta ligger på {matten, matta}."`

### Smart Tokenization

```python
def tokenize_preserve(text: str) -> List[str]:
    """Tokenize text while preserving exact spacing and punctuation."""
    return re.findall(r'\S+|\s+', text)
```

This gives us: `["Katta", " ", "ligger", " ", "på", " ", "matten", " ", "."]`

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

**Why POS tagging?** Because "kasta" could be:
- A **noun** (the act of throwing)
- A **verb** (to throw)

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
jenta: NOUN
kasta: VERB  
ballen: NOUN
```

## Stage 3: API Lookup Strategy

This is where things get sophisticated. We need to find ALL possible forms of a word.

### The Multi-Lemma Challenge

Here's a real example: "ballen" (the ball) has THREE different lemmas in the Norwegian dictionary:

1. **Lemma 5055**: ball (masculine) → ball, ballen, baller, ballene
2. **Lemma 159269**: ball (feminine) → balla, ballen, ballene
3. **Lemma 159267**: ball (neuter) → ball, baller, ballene

**The problem**: If we only looked at the first lemma, we'd miss "balla" as an alternative. But if we combine all lemmas blindly, we get irrelevant forms.

### Smart Lemma Selection

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

**Why this approach?** We only use lemmas that actually contain the word we're analyzing. This prevents "ballen" from getting random feminine forms that don't apply.

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
- First run of "Jenta kasta ballen": ~2-3 seconds
- With caching: subsequent runs are ~100ms

**Cache key strategy**: `lemmas_jenta_nob_VERB` ensures we cache different results for the same word with different POS tags.

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

**Example for "ballen"**:
- Found inflection: `word_form='ballen', tags=('Sing', 'Def')`
- This means "ballen" is singular definite
- So we only want alternatives that are also singular definite

**Example output**:
```
🏷️ FINDING TAGS FOR: ballen
   Found match: ballen -> ('Sing', 'Def')
   Final matching tags: {('Sing', 'Def')}

🔍 COLLECTING ALTERNATIVES WITH MATCHING TAGS:
   ✅ ballen (tags: ('Sing', 'Def'))  
   ✅ balla (tags: ('Sing', 'Def'))   # Same grammatical slot!
   ❌ ball (tags: ('Sing', 'Ind')) - doesn't match  # Wrong form
```

## Stage 5: BERT-Based Acceptability Filtering

This is the sophisticated part. Not all grammatically correct alternatives make sense in context.

### The Context Problem

Consider: `"Hinduen syntes den kasta han var i var grei"`

- "kasta" could theoretically be "kastene" (grammatically valid)
- But "kastene" (plural) makes no sense with "den" (singular)

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
Original: "Hinduen syntes den kasta han var i var grei"
Masked:   "Hinduen syntes den [MASK] han var i var grei"

BERT scores:
- kasta: logit=7.521 (very likely)
- kastene: logit=1.497 (unlikely - doesn't fit with "den")
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
🔍 ANALYZING: kasta (position 17)
Context: Jenta kasta ballen til gutten. Hinduen syntes den [kasta] han var i var grei.

kasta    : Logit 7.521 (ORIGINAL)
kastene  : Logit 1.497
❌ REJECTING kastene: logit diff +6.023 > 2.0 (too improbable)
```

### Position Matters!

The same word can have different alternatives in different positions:

```python
# Position 3: "Jenta [kasta] ballen"  
kasta: acceptable alternatives = {kasta, kastet}

# Position 17: "syntes den [kasta] han var"
kasta: acceptable alternatives = {kasta} only
```

**Why?** Because the grammatical context is different. In position 3, past tense works. In position 17, it's part of a complex construction where only one form fits.

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

**Why this matters**: If the input is "Jenta", the output should be "{Jenta, Jenten}", not "{jenta, jenten}".

### Smart Ordering

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
🎯 PROCESSING: Jenta kasta ballen.

📝 WORDS: ['jenta', 'kasta', 'ballen']

🏷️ POS TAGS:
   jenta: NOUN
   kasta: VERB
   ballen: NOUN

📡 API LOOKUP: jenta (POS: NOUN)
💾 CACHE HIT: lemmas for 'jenta' (POS: NOUN)
   ✅ jenta: 2 alternatives: ['jenta', 'jenten']

🧠 ACCEPTABILITY FILTERING (threshold: 3.00)

🔍 ANALYZING: jenta (position 0)
   Context: [Jenta] kasta ballen.
   Alternatives: ['jenta', 'jenten']
     jenten      : Logit 3.017, Prob 7.56e-05, Rank -1
     jenta       : Logit 3.155, Prob 2.83e-05, Rank -1 (ORIGINAL)
     ✅ KEEPING jenten: logit diff +0.138 <= 3.0
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
- **Solution**: File-based cache with smart key generation
- **Result**: 10x speed improvement on subsequent runs

## Summary

AltMorph is a sophisticated system that combines:
1. **Norwegian POS tagging** for linguistic accuracy
2. **Comprehensive API querying** for completeness  
3. **BERT contextual scoring** for acceptability
4. **Smart caching and concurrency** for performance
5. **Robust error handling** for reliability

The key insight is that morphological alternatives aren't just about grammar—context matters enormously. A word that's grammatically correct might be contextually inappropriate, and only advanced language models can make these subtle distinctions.

The system prioritizes correctness over speed, but uses caching and concurrency to make it practical for real-world use. Every design decision was made to handle the complexity of natural language while providing reliable, fast results.

When you see the output `"{Jenta, Jenten} {kasta, kastet} ballen til gutten."`, you're seeing the result of a complex pipeline that considered grammar, context, and acceptability to give you only the alternatives that truly make sense.
