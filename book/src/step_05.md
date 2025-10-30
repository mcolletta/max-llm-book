# Step 05: Token embeddings

<div class="note">
    Learn to create token embeddings that convert discrete token IDs into continuous vector representations.
</div>

## What are token embeddings?

In this section you will create the `Embedding` class. This converts discrete token IDs (integers) into continuous vector representations that the model can process.

The embedding layer is a lookup table with shape [50257, 768] where 50257 is GPT-2's vocabulary size and 768 is the embedding dimension. When you pass in token ID 1000, it returns row 1000 as the embedding vector.

These learned embeddings capture semantic relationships—similar words end up with similar vectors during training.

## Why use token embeddings?

**1. Continuous Representation**: Neural networks operate on continuous values, not discrete symbols. Token embeddings convert discrete token IDs (integers from 0 to vocab_size-1) into dense vectors that can be processed by matrix operations, allowing the model to learn meaningful transformations.

**2. Semantic Relationships**: During training, token embeddings naturally cluster semantically similar words closer together in vector space. Words like "king" and "queen" end up with similar embedding vectors, while unrelated words like "king" and "bicycle" are far apart. This learned structure is crucial for the model's understanding.

**3. Dimensionality Control**: Raw one-hot encoded tokens would be 50,257-dimensional sparse vectors. Token embeddings compress this to a dense 768-dimensional representation, making computation tractable while preserving (and enhancing) the information needed for language understanding.

**4. Shared Representation**: The same embedding vectors are used regardless of a token's position in the sequence. This parameter sharing reduces model size and allows the model to generalize patterns learned from one context to other contexts.

### Key concepts

**Embedding Lookup Table**:
- Stores one vector per vocabulary token
- Shape: [vocab_size, embedding_dim]
- Token ID `i` maps to row `i` of the table
- Initialized randomly, then learned during training

**MAX Embedding API**:
- [`Embedding(num_embeddings, dim)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Embedding): Creates embedding lookup table
- `num_embeddings`: Size of vocabulary (50257 for GPT-2)
- `dim`: Embedding dimension (768 for GPT-2 base)
- Automatically initializes weights with proper initialization scheme

**GPT-2 Vocabulary**:
- 50,257 unique tokens (byte-pair encoding)
- Includes common words, subwords, and special tokens
- Same vocabulary used by HuggingFace's GPT-2 implementation
- Token IDs range from 0 to 50256

**Embedding Dimension**:
- GPT-2 base uses 768 dimensions
- Larger models use 1024 (medium), 1280 (large), or 1600 (XL)
- Must match throughout the architecture
- Referred to as `n_embd` in the config

**HuggingFace Naming Convention**:
- `wte` stands for "word token embeddings"
- Matches the naming in original GPT-2 code
- Important for loading pretrained weights correctly

### Implementation tasks (`step_05.py`)

1. **Import Required Modules** (Lines 12-13):
   - Import `Embedding` from `max.nn.module_v3`
   - Import `Module` from `max.nn.module_v3`
   - Config is already imported for you

2. **Create Token Embedding Layer** (Lines 24-26):
   - Use `Embedding(config.vocab_size, dim=config.n_embd)`
   - `config.vocab_size` is 50257 (GPT-2's vocabulary)
   - `dim=config.n_embd` is 768 (embedding dimension)
   - Store in `self.wte`

3. **Implement Forward Pass** (Lines 39-41):
   - Call `self.wte(input_ids)` to lookup embeddings
   - Input: token IDs of shape [batch_size, seq_length]
   - Output: embeddings of shape [batch_size, seq_length, n_embd]
   - Return the result directly

**Implementation**:

```python
{{#include ../../steps/step_05.py}}
```

### Validation

Run `pixi run s05`

**Reference**: `solutions/solution_05.py`

---

**Next**: In [Step 06](./step_06.md), you'll implement position embeddings to encode sequence order information, which will be combined with these token embeddings.
