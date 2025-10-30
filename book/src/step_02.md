# Step 02: Causal masking

<div class="note">
    Learn to create attention masks to prevent the model from "seeing" future tokens during autoregressive generation.
</div>

## What is causal masking?

In this section you will implement the `causal_mask()` function. This creates a [mask matrix](https://docs.modular.com/glossary/ai/attention-mask/) that prevents the model from "seeing" future tokens when predicting the next token.

The mask sets attention scores to negative infinity (`-inf`) for future positions. After using `softmax()`, these `-inf` values become zero probability, blocking information flow from tokens that come later in the sequence.

This creates a lower triangular pattern where each token can only attend to itself and previous tokens.

## Why use causal masking?

**1. Autoregressive Generation**: GPT-2 generates text one token at a time, left-to-right. During training, causal masking prevents the model from "cheating" by looking ahead at tokens it should be predicting, forcing it to learn genuine next-token prediction.

**2. Training-Inference Consistency**: Without causal masking during training, the model would have access to information it won't have during actual text generation, creating a train-test mismatch that degrades performance.

**3. Parallel Training**: Causal masking enables efficient parallel training. Instead of computing predictions sequentially, we can process an entire sequence at once, with the mask ensuring each position only uses appropriate context.

**4. KV Cache Compatibility**: The causal structure allows for key-value caching during generation - we can cache past token representations and only compute new ones, significantly speeding up inference.

### Key concepts

**Causal Masking Mechanics**:
- Sets attention scores to `-inf` for future positions before softmax
- After softmax, `-inf` values become 0 probability (no attention)
- Creates lower triangular attention pattern: position `i` can attend to positions `0` through `i`
- Essential for autoregressive language modeling

**MAX Functional Programming**:
- [`@F.functional`](https://docs.modular.com/max/api/python/experimental/functional/#max.experimental.functional.functional) decorator converts functions to graph operations
- Enables type flexibility and optimization across different tensor types
- Required for functions that will be traced and compiled by MAX

**MAX Tensor Operations**:
- [`Tensor.constant()`](https://docs.modular.com/max/api/python/experimental/tensor#max.experimental.tensor.Tensor.constant): Creates a scalar constant tensor with specified dtype and device
- [`F.broadcast_to()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.broadcast_to): Expands tensor dimensions to match target shape
- [`F.band_part()`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.band_part): Extracts band matrix (keeps central diagonal band, zeros out rest)

**Mask Shape and Dimensions**:
- Input: `sequence_length` (current sequence) and `num_tokens` (additional context)
- Output shape: `(sequence_length, sequence_length + num_tokens)`
- Allows attending to both current sequence and previous context (KV cache)

### Implementation tasks (`step_02.py`)

1. **Import Required Modules** (Lines 1-10):
   - `Device` from `max.driver` - specifies hardware device (CPU/GPU)
   - `DType` from `max.dtype` - data type specification
   - `functional as F` from `max.experimental` - functional operations library
   - `Tensor` from `max.experimental.tensor` - tensor operations
   - `Dim`, `DimLike` from `max.graph` - dimension handling

2. **Add @F.functional Decorator** (Line 13):
   - Converts the function to a MAX graph operation
   - Required for compilation and optimization

3. **Calculate Total Sequence Length** (Line 34):
   - Combine `sequence_length` and `num_tokens` using `Dim()`
   - This determines the width of the attention mask

4. **Create Constant Tensor** (Lines 36-40):
   - Use `Tensor.constant(float("-inf"), dtype=dtype, device=device)`
   - Creates a scalar tensor that will be broadcast
   - `-inf` values will become 0 after softmax, blocking attention

5. **Broadcast to Target Shape** (Lines 42-46):
   - Use `F.broadcast_to(mask, shape=(sequence_length, n))`
   - Expands the scalar to a 2D matrix
   - Shape: `[sequence_length, sequence_length + num_tokens]`

6. **Apply Band Part** (Lines 48-52):
   - Use `F.band_part(mask, num_lower=None, num_upper=0, exclude=True)`
   - `num_lower=None`: Keep all diagonals below main diagonal (past tokens)
   - `num_upper=0`: Keep main diagonal (current token)
   - `exclude=True`: Set band to 0, outside band to original values (inverted behavior)
   - Creates lower triangular pattern with 0s on/below diagonal, `-inf` above

**Implementation**:

```python
{{#include ../../steps/step_02.py}}
```

### Validation

Run `pixi run s02`

**Reference**: `solutions/solution_02.py`

**Next**: In [Step 03](./step_03.md), you'll implement layer normalization to stabilize activations for effective training.