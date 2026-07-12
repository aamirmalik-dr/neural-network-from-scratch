# The math behind nnscratch

This document derives every gradient the library computes, in the same
notation and layer order as the code. If you read this next to
`src/nnscratch/layers.py` and `src/nnscratch/losses.py`, each formula maps to
one or two lines of NumPy.

## Notation

A minibatch has `n` rows. For one Dense layer,

- input `X` with shape `(n, d_in)`, one sample per row
- weights `W` with shape `(d_in, d_out)`, bias `b` with shape `(d_out,)`
- output `Z = X W + b` with shape `(n, d_out)`

The scalar loss is `L`. For any tensor `T`, I write `dT` for `dL/dT`, the
array of partial derivatives of `L` with respect to every entry of `T`. Each
`dT` has the same shape as `T`, which is a useful sanity check on every line
below.

## 1. The forward pass

A feedforward classifier is a composition of simple functions:

```
X -> Dense -> ReLU -> Dense -> ReLU -> Dense -> logits -> softmax + cross entropy -> L
```

Each layer only needs to know how to map its input to its output and to cache
whatever the backward pass will need. `Dense` caches `X`, `ReLU` caches the
sign mask, `Sigmoid` caches its output, `Dropout` caches its mask.

## 2. Backpropagation is the chain rule, organized

For a composition `L = f_k(f_{k-1}(... f_1(X)))`, the chain rule says the
gradient of `L` with respect to any intermediate quantity factors through the
layers above it. Backpropagation evaluates those factors from the top down.
Every layer implements one contract:

> Given `dZ`, the gradient of the loss with respect to my output, return
> `dX`, the gradient with respect to my input, and store the gradients with
> respect to my own parameters.

The `Sequential` container calls `backward` on each layer in reverse order,
feeding each layer's `dX` in as the next layer's `dZ`. No layer needs to know
what sits above or below it.

## 3. Softmax fused with cross entropy

For one sample with logits `z` (a row of length `C`) and true class `y`, the
softmax probability of class `j` and the cross entropy loss are

```
p_j = exp(z_j) / sum_k exp(z_k)
L   = -log(p_y)
```

Differentiating `L = -z_y + log(sum_k exp(z_k))` with respect to `z_j`:

```
dL/dz_j = -1[j = y] + exp(z_j) / sum_k exp(z_k) = p_j - 1[j = y]
```

So the gradient of the fused block is simply `p - onehot(y)`. That is the
whole reason to fuse: differentiating softmax alone produces a `C x C`
Jacobian per sample, while the fused gradient is one subtraction. Over a
batch the loss is the mean, so the code returns `(P - onehot(Y)) / n`.

For numerical stability the code computes log probabilities with the
log-sum-exp shift `z - max(z)`, which changes nothing mathematically because
softmax is shift invariant.

## 4. The Dense layer, three gradients

Forward: `Z = X W + b`. Given `dZ` of shape `(n, d_out)`, I need `dW`, `db`,
and `dX`.

Entry by entry, `Z_{ij} = sum_k X_{ik} W_{kj} + b_j`. Then:

**Weights.** `W_{kj}` touches `Z_{ij}` for every sample `i`, with coefficient
`X_{ik}`:

```
dW_{kj} = sum_i X_{ik} dZ_{ij}      =>      dW = X^T dZ        shape (d_in, d_out)
```

**Bias.** `b_j` shifts `Z_{ij}` for every sample, coefficient 1:

```
db_j = sum_i dZ_{ij}                =>      db = dZ.sum(axis=0)  shape (d_out,)
```

**Input.** `X_{ik}` touches `Z_{ij}` for every output unit `j`, coefficient
`W_{kj}`:

```
dX_{ik} = sum_j dZ_{ij} W_{kj}      =>      dX = dZ W^T          shape (n, d_in)
```

The shapes force the formulas: `X^T dZ` is the only way to multiply those two
arrays into the shape of `W`, and the same holds for `dX`. This is how I
check the code without redoing the index algebra.

**L2 regularization.** With the penalty `0.5 * l2 * ||W||^2` added to the
loss, its gradient `l2 * W` is added to `dW` inside `Dense.backward`, and the
penalty itself is added to the reported loss. Keeping both sides consistent
matters: the gradient check differentiates the total loss numerically, so it
verifies the regularizer too.

## 5. Activations

Elementwise activations have diagonal Jacobians, so backward is an
elementwise product with the local derivative.

**ReLU.** `a = max(0, z)`, derivative `1[z > 0]`:

```
dZ = dA * (Z > 0)
```

**Sigmoid.** `s(z) = 1 / (1 + exp(-z))`. Writing the derivative in terms of
the output, `s'(z) = s(z)(1 - s(z))`, so caching the forward output is enough:

```
dZ = dA * s * (1 - s)
```

The forward pass evaluates the sigmoid piecewise (one branch for positive
inputs, one for negative) so `exp` never overflows.

**Inverted dropout.** During training, a mask `M` zeroes each unit with
probability `rate` and scales survivors by `1 / (1 - rate)`, keeping the
expected activation unchanged. The layer is linear in its input given the
mask, so the backward pass reuses the same mask: `dZ = dA * M`. At inference
the layer is the identity.

## 6. The optimizer update rules

After backpropagation, every parameter `p` has a gradient `g`. The three
implemented optimizers differ only in how they turn `g` into a step.

**SGD.**

```
p <- p - lr * g
```

**SGD with momentum.** A velocity accumulates an exponentially weighted sum
of past gradients, which damps oscillation across steep directions and
accelerates along consistent ones:

```
v <- mu * v - lr * g
p <- p + v
```

**Adam.** Per-parameter first and second moment estimates with bias
correction. The effective step size adapts to the gradient scale:

```
m <- beta1 * m + (1 - beta1) * g
v <- beta2 * v + (1 - beta2) * g^2
m_hat = m / (1 - beta1^t),   v_hat = v / (1 - beta2^t)
p <- p - lr * m_hat / (sqrt(v_hat) + eps)
```

The bias correction matters early: at `t = 1`, `m = (1 - beta1) g`, and
dividing by `1 - beta1` recovers `g`, so the first step already has magnitude
close to `lr` regardless of gradient scale. `tests/test_optimizers.py` checks
exactly this property.

## 7. Initialization

For a Dense layer with `fan_in` inputs, the variance of a preactivation is
roughly `fan_in * Var(w) * Var(x)`. Keeping that variance stable across
layers prevents activations from exploding or vanishing with depth.

- **He (Kaiming) normal**, `Var(w) = 2 / fan_in`: the factor 2 compensates
  for ReLU zeroing half of its inputs. Used before ReLU layers.
- **Xavier (Glorot) normal**, `Var(w) = 2 / (fan_in + fan_out)`: balances the
  forward and backward variances. Used before symmetric activations and for
  the output layer.

## 8. Gradient checking, the correctness proof

Every derivation above can be verified mechanically. For a scalar loss `L(p)`
and any single parameter entry `p_i`, the central difference

```
dL/dp_i  approx  (L(p_i + eps) - L(p_i - eps)) / (2 eps)
```

has error `O(eps^2)`. The checker in `src/nnscratch/gradient_check.py`
perturbs every entry of every parameter array, rebuilds the numerical
gradient, and compares it to the backpropagation result with the symmetric
relative error

```
||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric|| + tiny)
```

With float64 and `eps = 1e-6`, a correct implementation lands around 1e-8 or
below; a wrong gradient shows up around 1e-2 or worse, so the two outcomes
are unambiguous. The test suite (`tests/test_gradient_check.py`) enforces
1e-5 across per-layer checks and whole-network checks with ReLU, Sigmoid,
depth, and L2 active. Two caveats worth knowing:

- ReLU is not differentiable at exactly 0, so per-layer checks keep inputs
  away from the kink.
- Dropout is stochastic, so gradient checks run with `training=False`, where
  the network is deterministic.
