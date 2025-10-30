# Step 07: Query/Key/Value projections (single head)

<div class="note">
    Learn to implement Q/K/V projection layers that transform embeddings for attention computation.
</div>

## What are query/key/value projections?

In this section you will implement the Q/K/V projections for attention. These linear layers transform input embeddings into three different representations:
- **Query**: "what am I looking for?"
- **Key**: "what do I contain?"
- **Value**: "what information do I carry?"

GPT-2 uses a single combined linear layer called `c_attn` that projects from embedding dimension (768) to 3× that size (2304). The output is then split into separate Q, K, and V tensors. This is more efficient than three separate layers.

## Why use Q/K/V projections?

**1. Learned Attention Patterns**: Without projections, attention would be limited to comparing embeddings directly using their original representations. Projections allow the model to learn transformations that make certain patterns easier to detect. For example, the model might learn to project "subject" tokens similarly in query space and "verb" tokens similarly in key space, making subject-verb relationships easier to capture.

**2. Flexible Representations**: The same input embedding gets projected into three different spaces (Q, K, V), each optimized for its role. The query projection learns "how to ask questions about context," the key projection learns "how to advertise what information is available," and the value projection learns "what information to pass forward." This flexibility allows the model to use different aspects of the same token for different purposes.

**3. Separation of Matching and Content**: Separating keys (used for matching) from values (used for content) is crucial. The model can learn that token A should attend to token B (based on Q-K similarity) while extracting different information from B's value. For instance, when processing "The cat sat on the mat," the token "sat" might attend to "cat" to understand the subject, but the value from "cat" provides semantic information rather than just the matching signal.

**4. Multi-Head Preparation**: Q/K/V projections enable multi-head attention (covered in Step 09). Each attention head gets a portion of the projected Q, K, V dimensions, allowing different heads to learn different attention patterns—some might focus on positional relationships, others on semantic similarity, and others on syntactic structure. The projections provide the raw material that gets divided among heads.

### Key concepts

**Linear Projections**:
- Transform input: `[batch, seq_length, n_embd]` → `[batch, seq_length, n_embd]`
- Implemented with [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear)
- Each projection learns a weight matrix and bias vector
- Same operation applied independently to each token's embedding

**HuggingFace Combined Projection**:
- Single layer projects to 3× embedding dimension: `Linear(n_embd, 3 * n_embd)`
- Named `c_attn` (combined attention projection)
- More efficient than three separate projections
- Output is split into Q, K, V after projection

**Splitting with F.split**:
- [`F.split(tensor, split_sizes, axis)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.split) divides tensor along an axis
- Split concatenated Q/K/V into three equal parts
- Each part has shape `[batch, seq_length, n_embd]`
- `split_sizes=[n_embd, n_embd, n_embd]` specifies three equal chunks

**Single Head Simplification**:
- Full GPT-2 uses 12 attention heads (768 dims ÷ 12 = 64 dims per head)
- This step uses a single head with full 768 dimensions
- Simpler to understand before introducing head splitting
- Step 09 will extend to multi-head attention

**Bias Parameters**:
- `bias=True` adds learnable bias to each projection
- Matches GPT-2 architecture (biases are used)
- Allows projections to shift outputs, not just rotate/scale

### Implementation tasks (`step_07.py`)

1. **Import Required Modules** (Lines 13-15):
   - Import `Linear` and `Module` from `max.nn.module_v3`
   - Import `functional as F` from `max.experimental`
   - Config is already imported for you

2. **Create Combined Q/K/V Projection** (Lines 27-30):
   - Use `Linear(config.n_embd, 3 * config.n_embd, bias=True)`
   - `config.n_embd` is 768 (input dimension)
   - `3 * config.n_embd` is 2304 (output for concatenated Q, K, V)
   - Store in `self.c_attn` (combined attention projection)

3. **Project Input to Q/K/V** (Lines 45-46):
   - Call `self.c_attn(x)` to project input
   - Input shape: `[batch, seq_length, n_embd]`
   - Output shape: `[batch, seq_length, 3 * n_embd]`
   - Store in `qkv` variable

4. **Split into Separate Q, K, V** (Lines 49-51):
   - Use `F.split(qkv, [self.n_embd, self.n_embd, self.n_embd], axis=-1)`
   - Splits last dimension into three equal parts
   - Each part has shape `[batch, seq_length, n_embd]`
   - Returns tuple of `(query, key, value)`

**Implementation**:

```python
{{#include ../../steps/step_07.py}}
```

### Validation

Run `pixi run s07`

**Reference**: `solutions/solution_07.py`

---

**Next**: In [Step 08](./step_08.md), you'll implement the attention mechanism itself, computing attention scores from Q and K, applying causal masking, and using those scores to weight the values.
