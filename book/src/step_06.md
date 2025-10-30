# Step 06: Position embeddings

<div class="note">
    Learn to create position embeddings that encode sequence order information for the model.
</div>

## What are position embeddings?

In this section you will create position embeddings to encode where each token appears in the sequence. While token embeddings tell the model "what" each token is, position embeddings tell it "where" the token is located.

Position embeddings work like token embeddings: a lookup table with shape [1024, 768] where 1024 is the maximum sequence length. Position 0 gets the first row, position 1 gets the second row, and so on.

These position vectors are added to token embeddings before entering the transformer blocks. Without position information, "dog bites man" and "man bites dog" would look identical to the model.

## Why use position embeddings?

**1. Sequence Order Matters**: Language is inherently sequential—word order determines meaning. The sentence "not good" has opposite meaning from "good." Since transformers process all positions in parallel through attention, they need explicit position information to distinguish token order. Position embeddings provide this crucial ordering signal.

**2. Parallel Processing Requirement**: Unlike RNNs that process sequences sequentially (inherently encoding position through time steps), transformers attend to all positions simultaneously. This parallelism enables much faster training but loses positional information. Position embeddings restore this information without sacrificing the parallel processing advantage.

**3. Long-Range Dependencies**: Position embeddings help the model learn position-dependent patterns. For instance, the model can learn that question marks typically appear at specific positions relative to question words, or that certain phrase structures appear at particular sentence positions. This position awareness improves the model's ability to capture syntactic and structural patterns.

**4. Complementary to Attention**: While attention mechanisms learn "what to attend to" based on content similarity, position embeddings help the model learn "where to attend" based on position relationships. Queries like "attend to the previous token" or "attend to tokens 5 positions away" become learnable patterns through the combination of content embeddings and position embeddings.

### Key concepts

**Position Indices**:
- Sequence positions: 0, 1, 2, ..., seq_length-1
- Created with [`Tensor.arange(seq_length, dtype, device)`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.arange)
- Must match the device and dtype of input tokens
- Independent of batch size (same positions used for all examples)

**MAX Embedding for Positions**:
- Same `Embedding` class as token embeddings
- [`Embedding(num_embeddings, dim)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Embedding)
- For positions: `num_embeddings = n_positions` (1024 for GPT-2)
- Embedding dimension matches token embeddings (768)

**Maximum Sequence Length**:
- GPT-2 supports up to 1024 tokens per sequence
- Position embeddings trained only up to this length
- Cannot process sequences longer than `n_positions` without modifications
- Referred to as `n_positions` in the config

**HuggingFace Naming Convention**:
- `wpe` stands for "word position embeddings"
- Matches original GPT-2 implementation
- Parallel to `wte` (word token embeddings)
- Essential for loading pretrained weights

**Adding Embeddings**:
- Token embeddings + position embeddings (element-wise addition)
- Both have shape [batch, seq_length, n_embd]
- Combined before entering transformer blocks
- Will see this combination in later steps

### Implementation tasks (`step_06.py`)

1. **Import Required Modules** (Lines 13-15):
   - Import `Tensor` from `max.experimental.tensor` (for Tensor.arange)
   - Import `Embedding` from `max.nn.module_v3`
   - Import `Module` from `max.nn.module_v3`
   - Config is already imported for you

2. **Create Position Embedding Layer** (Lines 27-29):
   - Use `Embedding(config.n_positions, dim=config.n_embd)`
   - `config.n_positions` is 1024 (maximum sequence length)
   - `dim=config.n_embd` is 768 (embedding dimension)
   - Store in `self.wpe` (word position embeddings)

3. **Implement Forward Pass** (Lines 42-44):
   - Call `self.wpe(position_ids)` to lookup position embeddings
   - Input: position indices of shape [seq_length] or [batch, seq_length]
   - Output: position embeddings of shape [seq_length, n_embd] or [batch, seq_length, n_embd]
   - Return the result directly

**Implementation**:

```python
{{#include ../../steps/step_06.py}}
```

### Validation

Run `pixi run s06`

**Reference**: `solutions/solution_06.py`

---

**Next**: In [Step 07](./step_07.md), you'll begin implementing the attention mechanism, starting with Query/Key/Value projections that use both the token and position embeddings you've created.
