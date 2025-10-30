# Step 04: Feed-forward network

<div class="note">
    Learn to build the feed-forward network (MLP) that processes information after attention in each transformer block.
</div>

## What is the GPT-2 MLP?

In this section you will build the `MLP` class. This is a two-layer feed-forward network that appears after the attention mechanism in every transformer block.

The MLP expands the embedding dimension by 4x (768 → 3072), applies a GELU activation function, then projects back to the original dimension (3072 → 768). This "expansion and contraction" pattern processes each token position independently, adding non-linear transformations to the attention outputs.

## Why use an MLP in transformers?

**1. Non-Linear Transformations**: While attention provides a powerful mechanism for aggregating information across tokens, it's fundamentally a linear operation (weighted sum). The MLP adds crucial non-linearity through the GELU activation function, enabling the model to learn complex patterns.

**2. Position-Wise Processing**: The MLP processes each position independently (unlike attention which looks across positions), allowing the model to refine and transform the attended representations at each position.

**3. Capacity and Expressiveness**: The intermediate layer expansion (typically 4x the embedding dimension in GPT-2) provides additional capacity for the model to learn rich transformations. This expansion is critical for model performance.

**4. Information Mixing**: While attention mixes information across sequence positions, the MLP mixes information across feature dimensions at each position, providing a complementary form of computation.

### Key concepts

**MLP Architecture**:
- Two linear layers: `c_fc` (expansion) and `c_proj` (projection)
- `c_fc`: Projects from embedding dimension (768) to intermediate size (typically 3072 = 4�768)
- `c_proj`: Projects from intermediate size back to embedding dimension
- Non-linear activation (GELU) between the two layers
- Both layers use bias terms

**GELU Activation Function**:
- GELU (Gaussian Error Linear Unit) is the activation function used in GPT-2
- Smoother alternative to ReLU, incorporating probabilistic behavior
- Mathematical formula:

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

- The `approximate="tanh"` parameter uses a faster tanh-based approximation:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right)\right)$$

- Provides smooth gradients and better training dynamics than ReLU

**MAX Linear Layers**:
- [`Linear(in_features, out_features, bias=True)`](https://docs.modular.com/max/api/python/nn/module_v3#max.nn.module_v3.Linear): Applies linear transformation `y = xW^T + b`
- `in_features`: Size of input feature dimension
- `out_features`: Size of output feature dimension
- `bias`: Whether to include a learnable bias term (GPT-2 uses bias=True)

**MAX GELU Function**:
- [`F.gelu(input, approximate="tanh")`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.gelu): Applies GELU activation
- `input`: Input tensor to transform
- `approximate`: Approximation method - "tanh" for faster computation (matches GPT-2), "none" for exact calculation

**Layer Naming Convention**:
- `c_fc`: "c" prefix is HuggingFace convention, "fc" stands for "fully connected" (the expansion layer)
- `c_proj`: "proj" stands for "projection" (projects back to embedding dimension)
- These names match the original GPT-2 checkpoint structure for weight loading compatibility

### Implementation tasks (`step_04.py`)

1. **Import Required Modules** (Lines 1-9):
   - `functional as F` from `max.experimental` - provides F.gelu() activation function
   - `Tensor` from `max.experimental.tensor` - tensor operations (used implicitly)
   - `Linear` from `max.nn.module_v3` - linear transformation layers
   - `Module` from `max.nn.module_v3` - base class for neural network modules

2. **Create First Linear Layer (c_fc)** (Lines 25-29):
   - Use `Linear(embed_dim, intermediate_size, bias=True)`
   - This is the expansion layer that increases dimensionality
   - Maps from embedding dimension (768) to intermediate size (typically 3072)
   - Stores in `self.c_fc`

3. **Create Second Linear Layer (c_proj)** (Lines 31-35):
   - Use `Linear(intermediate_size, embed_dim, bias=True)`
   - This is the projection layer that restores original dimensionality
   - Maps from intermediate size back to embedding dimension
   - Stores in `self.c_proj`

4. **Apply First Linear Transformation** (Lines 46-49):
   - Apply `self.c_fc(hidden_states)` to expand the representation
   - This transforms shape from `[batch, seq_len, embed_dim]` to `[batch, seq_len, intermediate_size]`
   - Reassign result to `hidden_states`

5. **Apply GELU Activation** (Lines 51-55):
   - Use `F.gelu(hidden_states, approximate="tanh")`
   - Applies non-linear transformation element-wise
   - The `approximate="tanh"` matches GPT-2's implementation for efficiency
   - Reassign result to `hidden_states`

6. **Apply Second Linear Transformation** (Lines 57-60):
   - Apply `self.c_proj(hidden_states)` to project back to original dimension
   - This transforms shape from `[batch, seq_len, intermediate_size]` back to `[batch, seq_len, embed_dim]`
   - Return the final result

**Implementation**:

```python
{{#include ../../steps/step_04.py}}
```

### Validation

Run `pixi run s04`

**Reference**: `solutions/solution_04.py`

**Next**: In [Step 05](./step_05.md), you'll implement token embeddings to convert discrete token IDs into continuous vector representations.
