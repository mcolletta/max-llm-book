# Step 10: Residual connections and layer normalization

<div class="note">
    Learn to implement residual connections and layer normalization to enable training deep transformer networks.
</div>

## What are residual connections and layer normalization?

In this section you will combine residual connections and layer normalization into a reusable pattern for transformer blocks.

**Residual connections** add the input directly to the output: `output = input + layer(input)`. This creates shortcuts that help gradients flow through deep networks during training.

**Layer normalization** normalizes activations across features for each position independently, then applies learned scale and shift parameters. This stabilizes training.

GPT-2 uses **pre-norm architecture**: layer norm is applied before each sublayer (attention or MLP), following the pattern `x = x + sublayer(layer_norm(x))`.

## Why use residual connections?

**1. Gradient Flow**: Deep networks suffer from vanishing gradients—gradients shrink exponentially as they backpropagate through many layers. Residual connections create direct paths for gradients to flow backward through the network. During backpropagation, the gradient of `output = input + layer(input)` includes a term from the identity path (`∂output/∂input` includes a +1), ensuring gradients can flow unimpeded even through very deep networks.

**2. Identity Initialization**: At initialization, a network with residual connections can learn the identity function easily. If a layer's weights are near zero, `layer(input) ≈ 0`, so `output ≈ input`. The network starts in a reasonable state where information passes through, and layers can gradually learn useful transformations. Without residual connections, random initialization often produces outputs unrelated to inputs, making early training unstable.

**3. Ensemble Effect**: Residual networks can be viewed as implicit ensembles. Each residual connection creates multiple paths through the network—some paths skip layers, others pass through them. The final output combines information from all these paths. This ensemble-like behavior improves robustness and generalization.

**4. Information Preservation**: In transformers processing sequences, residual connections ensure that positional and token information from embeddings is preserved throughout all layers. Without residuals, this information might be lost as it passes through multiple transformations. Residuals guarantee that the original embeddings can always influence the final output.

## Why use layer normalization?

**1. Training Stability**: Without normalization, activation distributions shift during training (internal covariate shift), forcing later layers to constantly adapt to changing inputs. Layer norm stabilizes these distributions, allowing consistent learning across all layers. This is especially important for transformers, which can have dozens of layers.

**2. Scale Invariance**: Layer norm makes the network less sensitive to the scale of parameters. Large weight values don't cause exploding activations because normalization rescales them. This allows using higher learning rates and more aggressive optimization, speeding up training.

**3. Batch Independence**: Unlike batch normalization (which normalizes across the batch dimension), layer norm normalizes each example independently. This means behavior is identical during training and inference, and the model works with any batch size, including batch size 1. This is crucial for autoregressive generation where you process one token at a time.

**4. Learned Adaptation**: The gamma (scale) and beta (shift) parameters allow the network to learn the optimal distribution for each layer. If complete normalization isn't beneficial, the network can learn gamma and beta values that partially or fully undo it. This flexibility is important—normalization is helpful, but the network needs control over the final distribution.

### Key concepts

**Layer Normalization Formula**:

$$\text{output} = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where:
- $\mu = \text{mean}(x)$ is the mean across the last dimension
- $\sigma^2 = \text{variance}(x)$ is the variance across the last dimension
- $\gamma$ is the learnable scale parameter (weight)
- $\beta$ is the learnable shift parameter (bias)
- $\epsilon$ prevents division by zero (typically 1e-5)

**MAX Layer Norm Implementation**:
- [`F.layer_norm(x, gamma, beta, epsilon)`](https://docs.modular.com/max/api/python/experimental/functional#max.experimental.functional.layer_norm)
- `gamma`: learnable scale parameter (initialized to 1)
- `beta`: learnable shift parameter (initialized to 0)
- Normalizes over the last dimension automatically

**Learnable Parameters**:
- `weight` (gamma): `Tensor.ones([dim])` - initialized to 1
- `bias` (beta): `Tensor.zeros([dim])` - initialized to 0
- These allow the network to learn optimal scaling and shifting

**Pre-norm Architecture**:
- GPT-2 uses the pre-norm pattern for residual connections:

$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

- Apply layer norm first, then sublayer, then add residual
- More stable than post-norm: $\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$

**Residual Addition**:
- Simple element-wise addition: `input + sublayer_output`
- Both tensors must have identical shapes
- No additional parameters needed—just addition

### Implementation tasks (`step_10.py`)

1. **Import Required Modules** (Lines 13-17):
   - Import `functional as F` from `max.experimental`
   - Import `Tensor` from `max.experimental.tensor`
   - Import `DimLike` from `max.graph`
   - Import `Module` from `max.nn.module_v3`

2. **Initialize LayerNorm Parameters** (Lines 33-38):
   - Create `self.weight`: `Tensor.ones([dim])`
   - Create `self.bias`: `Tensor.zeros([dim])`
   - Store `self.eps` for numerical stability

3. **Implement LayerNorm Forward Pass** (Lines 50-51):
   - Call `F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)`
   - Returns normalized tensor with same shape as input

4. **Create ResidualBlock LayerNorm** (Lines 68-69):
   - Initialize `self.ln = LayerNorm(dim, eps=eps)`
   - This will be used to normalize before sublayers

5. **Implement Residual Connection** (Lines 83-84):
   - Return `x + sublayer_output`
   - Simple addition creates the residual connection

6. **Implement apply_residual_connection** (Lines 97-98):
   - Return `input_tensor + sublayer_output`
   - Standalone function demonstrating the pattern

**Implementation**:

```python
{{#include ../../steps/step_10.py}}
```

### Validation

Run `pixi run s10`

**Reference**: `solutions/solution_10.py`

---

**Next**: In [Step 11](./step_11.md), you'll combine everything learned so far—multi-head attention, MLP, layer norm, and residual connections—into a complete transformer block, the fundamental building block of GPT-2.
