# Advanced Machine Learning — Viva Preparation Notes

> **Scope**: Core algorithm implementations only (excludes testing code).  
> **Format**: Theory → Code reference → How it is implemented.

---

## Table of Contents

- [Assignment 1 — Gradient Boosting Trees (GBT)](#assignment-1--gradient-boosting-trees-gbt)
- [Assignment 2 — Gaussian Processes (GP)](#assignment-2--gaussian-processes-gp)
- [Assignment 3 — Variational Autoencoders (VAE & CVAE)](#assignment-3--variational-autoencoders-vae--cvae)

---

# Assignment 1 — Gradient Boosting Trees (GBT)

**Source files**: `Assignment1/src/gbt/core.py`, `Assignment1/src/gbt/utils.py`  
**Reference**: Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2009), Ch. 10.

---

## 1.1 What Is Gradient Boosting?

Gradient boosting builds an additive model **stage-by-stage**:

```
F_M(x) = F_0(x) + ν·h_1(x) + ν·h_2(x) + … + ν·h_M(x)
```

Each new weak learner `h_m` is fitted not to the original targets, but to the **negative gradient** of the loss — the direction that most reduces the loss at the current ensemble prediction. This converts any differentiable loss into a boosting algorithm (Algorithm 10.4 from ESL).

---

## 1.2 Initialisation

### Theory
The model starts with the **best constant prediction** that minimises the training loss:

- **Regression (MSE)**: `F_0(x) = argmin_γ Σ L(y_i, γ)` → `mean(y)`.
- **Classification (Logistic)**: `F_0(x) = log(p̄ / (1 − p̄))`, where `p̄ = mean(y)` is the fraction of positive labels.

### Code — Regression (`core.py`, lines 127–129)
```python
# Step 1: Initialise f_0(x) = argmin_γ Σ L(y_i, γ) = mean(y) for MSE
self.f0_ = np.mean(y)
F_train = np.full(n_samples, self.f0_)   # Current predictions for every sample
```
`F_train` is the running sum that gets updated at every boosting iteration.

### Code — Classification (`core.py`, lines 273–279)
```python
# Step 1: Initialise f_0(x) = log(p/(1-p)), where p = mean(y)
p_init = np.mean(y)
p_init = np.clip(p_init, 1e-15, 1 - 1e-15)  # avoid log(0)
self.f0_ = np.log(p_init / (1 - p_init))      # log-odds initialisation

F_train = np.full(n_samples, self.f0_)         # raw log-odds predictions
```
The clipping prevents numerical blowup when all labels are 0 or 1.

---

## 1.3 Pseudo-Residuals (Negative Gradient)

### Theory
At each boosting stage `m`, compute the **negative gradient of the loss** w.r.t. the current prediction:

```
r_im = − ∂L(y_i, F_{m-1}(x_i)) / ∂F_{m-1}(x_i)
```

- **MSE loss** `L = 0.5(y − F)²`: `r_i = y_i − F_{m-1}(x_i)` (ordinary residuals).
- **Logistic loss** `L = −y·log(p) − (1−y)·log(1−p)`: `r_i = y_i − σ(F_{m-1}(x_i))`.

These are called *pseudo-residuals* because for MSE they literally are residuals, but for other losses they generalise that idea.

### Code — MSE Negative Gradient (`utils.py`, lines 25–32)
```python
def mse_negative_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    For L(y, f) = 0.5 * (y - f)^2:
    -∂L/∂f = y - f  (ordinary residuals)
    """
    return y_true - y_pred
```

### Code — Logistic Negative Gradient (`utils.py`, lines 61–74)
```python
def logistic_negative_gradient(y_true: np.ndarray, y_pred_raw: np.ndarray) -> np.ndarray:
    """
    Gradient: ∂L/∂F = -y + p,  so negative gradient = y - p
    where p = sigmoid(F)
    """
    p = sigmoid(y_pred_raw)
    return y_true - p    # "residuals" in probability space
```

### Code — Boosting Loop Usage (`core.py`, lines 147–148 for regression; 297–298 for classification)
```python
# (a) Compute pseudo-residuals (negative gradient)
residuals = mse_negative_gradient(y_sub, F_sub)   # or logistic_negative_gradient
```

---

## 1.4 Fitting the Weak Learner (Tree)

### Theory
A shallow decision tree is fitted to the pseudo-residuals `{(x_i, r_im)}`. The tree partitions the feature space into `J` disjoint leaf regions `R_{1m}, …, R_{Jm}`.

**Key detail**: Even for classification, a **regression tree** (`DecisionTreeRegressor`) is used, because pseudo-residuals are continuous values.

### Code (`core.py`, lines 151–156 for regression; 301–306 for classification)
```python
# (b) Fit tree to residuals
tree = DecisionTreeRegressor(
    max_depth=self.max_depth,          # controls J (number of leaves ≈ 2^max_depth)
    min_samples_leaf=self.min_samples_leaf,
    random_state=self.random_state
)
tree.fit(X_sub, residuals)
```
`DecisionTreeRegressor` is always used — even for classification — because it fits a continuous target (the pseudo-residuals).

---

## 1.5 Leaf (Region) Optimisation — Optimal Gamma (γ)

### Theory
After the tree partitions the space, the split values assigned by the tree to each leaf are **not optimal** for our actual loss (the tree was fit on residuals, not the loss). We re-compute the best constant within each leaf:

```
γ_jm = argmin_γ  Σ_{x_i ∈ R_jm} L(y_i,  F_{m-1}(x_i) + γ)
```

- **MSE**: Closed-form → `γ* = mean(y_i − F_{m-1}(x_i))` in that leaf.
- **Logistic**: No closed form → one-step **Newton-Raphson** approximation:
  `γ* = Σ r_i / Σ p_i(1 − p_i)` (numerator = sum of pseudo-residuals, denominator = sum of second derivatives).

### Code — MSE Optimal Gamma (`utils.py`, lines 35–43)
```python
def mse_optimal_gamma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """γ* = mean of residuals in the leaf."""
    residuals = y_true - y_pred
    return np.mean(residuals)
```

### Code — Logistic Optimal Gamma (`utils.py`, lines 77–109)
```python
def logistic_optimal_gamma(y_true, y_pred_raw, negative_gradients) -> float:
    """Newton-Raphson step:  γ* = Σ r_i / Σ p_i(1-p_i)"""
    p = sigmoid(y_pred_raw)
    w = p * (1 - p)            # Second derivatives (Hessian diagonal)
    w = np.clip(w, 1e-15, None)
    numerator   = np.sum(negative_gradients)
    denominator = np.sum(w)
    if denominator == 0:
        return 0.0
    return numerator / denominator
```

### Code — Loop Over Leaves (`core.py`, lines 160–167 for regression)
```python
leaf_indices_sub = tree.apply(X_sub)    # which leaf does each training point fall in?
unique_leaves    = np.unique(leaf_indices_sub)

gamma_map = {}
for leaf_id in unique_leaves:
    mask = (leaf_indices_sub == leaf_id)
    gamma_map[leaf_id] = mse_optimal_gamma(y_sub[mask], F_sub[mask])
```
`tree.apply(X)` returns the integer leaf ID for every sample — then we compute the optimal gamma for each leaf independently.

---

## 1.6 Model Update with Shrinkage (Learning Rate)

### Theory
The model is updated by:

```
F_m(x) = F_{m-1}(x) + ν · Σ_j γ_jm · I(x ∈ R_jm)
```

where `ν ∈ (0,1]` is the **learning rate** (shrinkage). A smaller `ν` requires more trees but improves generalisation by regularising the additive expansion.

### Code (`core.py`, lines 169–172 for regression)
```python
# (d) Update predictions with shrinkage
leaf_indices_train = tree.apply(X)   # leaf assignment for ALL training samples
update = np.array([gamma_map.get(leaf, 0.0) for leaf in leaf_indices_train])
F_train += self.learning_rate * update   # ν · γ applied to every sample
```
`gamma_map.get(leaf, 0.0)` handles the edge case where a test-time leaf ID was not seen during training (defaults to zero update).

---

## 1.7 Stochastic Boosting (Row Subsampling)

### Theory
At each iteration, only a random fraction `subsample` of training samples is used to compute residuals and fit the tree. This introduces variance reduction (like random forests) and speeds up training.

### Code (`core.py`, lines 79–86 base class; lines 141–145 in regression loop)
```python
# Base class method
def _subsample_indices(self, n_samples, rng):
    if self.subsample < 1.0:
        n_subsample = max(1, int(self.subsample * n_samples))
        indices = rng.choice(n_samples, size=n_subsample, replace=False)
        return np.sort(indices)
    else:
        return np.arange(n_samples)

# In the boosting loop:
indices = self._subsample_indices(n_samples, rng)
X_sub   = X[indices]     # subset for residuals + tree fitting
y_sub   = y[indices]
F_sub   = F_train[indices]
```
Note: After fitting on the subset, predictions are updated on the **full** `X` (line 170: `tree.apply(X)`), which is standard stochastic gradient boosting.

---

## 1.8 Prediction

### Theory
Final prediction accumulates contributions from all trees:

```
F_M(x) = F_0 + ν · Σ_{m=1}^{M}  γ_{j(x),m}
```

where `j(x)` is the leaf index for point `x` in tree `m`.

### Code — `_predict_raw` (`core.py`, lines 199–222 for regression)
```python
def _predict_raw(self, X, up_to_iteration=None):
    n_estimators = up_to_iteration if up_to_iteration else len(self.estimators_)
    F = np.full(X.shape[0], self.f0_)          # Start at constant baseline

    for m in range(n_estimators):
        tree      = self.estimators_[m]
        gamma_map = self.leaf_values_[m]
        leaf_indices = tree.apply(X)
        update = np.array([gamma_map.get(leaf, 0.0) for leaf in leaf_indices])
        F += self.learning_rate * update       # accumulate each tree's contribution

    return F
```

### Code — Classification probability (`core.py`, lines 376–387)
```python
def predict_proba(self, X):
    F = self._predict_raw(X)    # raw log-odds
    return sigmoid(F)           # convert to probability
```

---

## 1.9 Loss Functions Summary

| Loss | Formula | Negative Gradient | Leaf Optimum |
|---|---|---|---|
| MSE | `0.5(y−F)²` | `y − F` | `mean(residuals in leaf)` |
| Logistic | `−y·log(p)−(1−y)·log(1−p)` | `y − σ(F)` | `Σr_i / Σp_i(1−p_i)` (Newton step) |

### Code — Sigmoid helper (`utils.py`, lines 112–118)
```python
def sigmoid(x):
    """Numerically stable sigmoid — two branches to avoid overflow."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
```

---

## 1.10 Likely Viva Questions

| Question | Short Answer | Where in Code |
|---|---|---|
| Why use a regression tree for classification? | Pseudo-residuals are continuous — we need to fit a real value. | `core.py` line 301 |
| What is the role of learning rate? | Shrinkage — multiplies every leaf update, prevents overfitting. | `core.py` line 172 |
| What is the difference between γ for MSE vs logistic? | MSE: mean of residuals (exact). Logistic: Newton step (1st-order approx). | `utils.py` lines 35–109 |
| Why re-optimise leaf values after tree fitting? | Tree minimises squared residuals, not the actual loss — re-computing γ per leaf aligns with the true objective. | `core.py` lines 160–167 |
| What is stochastic boosting? | Row subsampling per iteration — reduces variance, regularises. | `core.py` lines 79–86 |
| What is stored in `leaf_values_`? | A list (one per tree) of `{leaf_id: γ}` dicts — the optimal constant per leaf. | `core.py` line 176 |

---
---

# Assignment 2 — Gaussian Processes (GP)

**Source files**: `Assignment2/src/gp/base.py`, `kernels.py`, `regression.py`, `classification.py`  
**Reference**: Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006), Algorithms 2.1, 3.1, 3.2.

---

## 2.1 What Is a Gaussian Process?

A Gaussian Process (GP) defines a **distribution over functions**: any finite collection of function values `f(x_1), …, f(x_n)` is jointly Gaussian:

```
f(x) ~ GP(m(x),  k(x, x'))
```

- `m(x)` — mean function (usually zero).
- `k(x, x')` — **kernel** (covariance function) encodes assumptions about smoothness.

GPs are non-parametric: the model complexity grows with data.

---

## 2.2 Kernels

A kernel `k(x, x')` measures similarity between inputs. The kernel matrix `K_{ij} = k(x_i, x_j)` must be **symmetric positive semi-definite**.

### 2.2.1 RBF (Squared Exponential) Kernel

**Theory**: `k(x, x') = σ² · exp(−‖x−x'‖² / (2ℓ²))`

Infinite differentiability. `ℓ` = length-scale (how quickly correlation decays), `σ²` = signal variance.

**Code** (`Assignment2/src/gp/kernels.py`, lines 86–101)
```python
class RBF(Kernel):
    def __call__(self, X1, X2=None, diag=False):
        dists_sq = self._squared_distances(X1, X2)
        K = self.variance * np.exp(-0.5 * dists_sq / (self.length_scale ** 2))
        return K

    def _squared_distances(self, X1, X2):
        # ||x-x'||² = ||x||² - 2<x,x'> + ||x'||²  (efficient, no loops)
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        return np.maximum(X1_sq + X2_sq.T - 2 * X1 @ X2.T, 0.0)
```

### 2.2.2 Matérn Kernel

**Theory**:
- ν = 3/2: `k(r) = σ²(1 + √3·r/ℓ)·exp(−√3·r/ℓ)` — once-differentiable.
- ν = 5/2: `k(r) = σ²(1 + √5·r/ℓ + 5r²/(3ℓ²))·exp(−√5·r/ℓ)` — twice-differentiable.

**Code** (`Assignment2/src/gp/kernels.py`, lines 183–190)
```python
if self.nu == 1.5:
    scaled = np.sqrt(3.0) * dists / self.length_scale
    K = self.variance * (1.0 + scaled) * np.exp(-scaled)
else:  # nu == 2.5
    scaled = np.sqrt(5.0) * dists / self.length_scale
    K = self.variance * (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)
```

### 2.2.3 Rational Quadratic Kernel

**Theory**: `k(x, x') = σ²(1 + ‖x−x'‖²/(2αℓ²))^{−α}` — infinite mixture of RBFs with different length-scales.

**Code** (`Assignment2/src/gp/kernels.py`, lines 268–271)
```python
dists_sq = self._squared_distances(X1, X2)
base = 1.0 + dists_sq / (2.0 * self.alpha * self.length_scale ** 2)
K = self.variance * (base ** (-self.alpha))
```

### 2.2.4 Hyperparameters in Log Space

All kernels store and optimise hyperparameters **in log space** (via `get_params` / `set_params`), enabling unconstrained gradient-based optimisation while guaranteeing positivity.

**Code** (`Assignment2/src/gp/kernels.py`, lines 111–118 for RBF)
```python
def get_params(self) -> np.ndarray:
    return np.array([np.log(self.length_scale), np.log(self.variance)])

def set_params(self, params: np.ndarray) -> None:
    self.length_scale = np.exp(params[0])   # ensures ℓ > 0
    self.variance     = np.exp(params[1])   # ensures σ² > 0
```

---

## 2.3 Numerically Stable Cholesky Decomposition

### Theory
GP inference requires computing `K^{-1}y`. Direct inversion is numerically unstable and O(n³). The Cholesky decomposition `K = LL^T` (L lower-triangular) is the standard approach:

1. `K = LL^T`
2. Solve `Lα' = y` (forward substitution)
3. Solve `L^Tα = α'` (backward substitution)

If `K` is near-singular, adding a small diagonal **jitter** `ε·I` restores positive definiteness.

### Code — Adaptive Jitter Cholesky (`base.py`, lines 39–54)
```python
def stable_cholesky(K, jitter=1e-6):
    jitter_levels = [jitter, 1e-5, 1e-4, 1e-3, 1e-2]
    for j in jitter_levels:
        try:
            L = np.linalg.cholesky(K + np.eye(K.shape[0]) * j)
            return L
        except np.linalg.LinAlgError:
            continue     # try larger jitter
    raise np.linalg.LinAlgError("Cholesky failed even with max jitter")
```
If the first jitter level fails (matrix not positive definite), progressively larger values are tried.

### Code — Cholesky Solve (`base.py`, lines 57–79)
```python
def cholesky_solve(L, b):
    """Solve K x = b given L s.t. K = LL^T."""
    y = np.linalg.solve(L,    b)    # forward: L y  = b
    x = np.linalg.solve(L.T,  y)   # backward: L^T x = y
    return x
```

---

## 2.4 GP Regression — Algorithm 2.1

### Theory
Given training data `(X, y)` with `y = f(X) + ε`, `ε ~ N(0, σ_n²I)`, the posterior is:

```
p(f_*|X, y, x_*) = N(f̄_*, V[f_*])
f̄_*  = K_*^T α,             where α = (K + σ_n²I)^{-1} y
V[f_*] = k(x_*,x_*) − v^T v,  where v = L^{-1} K_*
```

### Code — `_compute_alpha` (`regression.py`, lines 117–131) — Steps 1–3 of Alg. 2.1
```python
def _compute_alpha(self):
    n = self.X_train_.shape[0]
    K  = self.kernel(self.X_train_, self.X_train_)  # n×n kernel matrix
    K += np.eye(n) * self.noise                      # add noise: K + σ_n²I

    self.L_     = stable_cholesky(K, jitter=1e-6)   # Step 2: Cholesky K = LL^T
    self.alpha_ = cholesky_solve(self.L_, self.y_train_)  # Step 3: α = K^{-1}y
```

### Code — `predict` (`regression.py`, lines 133–191) — Steps 4–5 of Alg. 2.1
```python
def predict(self, X, return_std=False, return_cov=False):
    K_star = self.kernel(self.X_train_, X)         # Step 4: K_* = k(X_train, X_test)
    mean   = K_star.T.dot(self.alpha_)              # f̄_* = K_*^T α

    if return_std:
        v   = np.linalg.solve(self.L_, K_star)      # v = L^{-1} K_*
        var = self.kernel(X, X, diag=True) - np.sum(v**2, axis=0)  # k_** - v^T v
        var = np.maximum(var, 0.0)                  # numerical safety
        return mean, np.sqrt(var)

    if return_cov:
        v   = np.linalg.solve(self.L_, K_star)
        cov = self.kernel(X, X) - v.T.dot(v)
        cov = 0.5 * (cov + cov.T)                  # ensure exact symmetry
        return mean, cov
```

---

## 2.5 Log Marginal Likelihood (LML)

### Theory
Hyperparameters (`ℓ`, `σ²`, `σ_n²`) are chosen by maximising the **log marginal likelihood**:

```
log p(y|X,θ) = −½ y^T K^{-1} y  −  ½ log|K|  −  n/2 · log(2π)
              ─────────────────    ───────────    ──────────────
               data fit             complexity       constant
```

The data-fit term rewards fitting `y`, while the complexity term penalises overly complex `K`, providing automatic Occam's razor.

### Code (`base.py`, lines 82–124)
```python
def log_marginal_likelihood(y, K, L=None, jitter=1e-6):
    if L is None:
        L = stable_cholesky(K, jitter=jitter)

    alpha    = cholesky_solve(L, y)
    data_fit = -0.5 * np.dot(y, alpha)          # -½ y^T K^{-1} y

    log_det  = -np.sum(np.log(np.diag(L)))       # -½ log|K| = -Σ log(L_ii)
    const    = -0.5 * len(y) * np.log(2 * np.pi)

    return data_fit + log_det + const
```
`log|K| = 2·Σ log(L_ii)` because `|K| = |L|² = (∏ L_ii)²`.

---

## 2.6 Hyperparameter Optimisation (L-BFGS-B)

### Theory
Maximise `log p(y|X,θ)` w.r.t. `θ = (log ℓ, log σ², log σ_n²)` using L-BFGS-B. Multiple random restarts guard against local optima.

### Code (`regression.py`, lines 236–302)
```python
def _optimise_hyperparameters(self):
    def objective(params):          # minimise NEGATIVE LML
        self.kernel.set_params(params[:-1])    # update kernel hyperparams
        self.noise = np.exp(params[-1])        # update noise (positive by exp)
        self._compute_alpha()                  # recompute L and α
        lml = self.log_marginal_likelihood_value()
        return -lml                            # negate for minimisation

    initial_params = np.concatenate([self.kernel.get_params(), [np.log(self.noise)]])
    bounds = [(-10, 10) for _ in initial_params]    # log-space bounds

    for restart in range(self.n_restarts + 1):
        x0 = initial_params if restart == 0 else np.random.uniform(-2, 2, ...)
        result = minimize(objective, x0=x0, method='L-BFGS-B', bounds=bounds)
        ...  # keep best result
```

---

## 2.7 GP Classification — Laplace Approximation

### Theory
For classification, the likelihood `p(y|f)` is non-Gaussian (logistic or probit), making exact inference intractable. The **Laplace approximation** fits a Gaussian `q(f|X,y) = N(f | f̂, (K^{-1} + W)^{-1})` at the posterior mode `f̂`.

**Algorithm 3.1 — Newton's Method for Posterior Mode:**
1. Initialise `f = 0`.
2. Repeat:
   - `W = diag(−∇∇ log p(y|f))` (negative Hessian, diagonal)
   - `b = Wf + ∇ log p(y|f)`
   - `B = I + W^{1/2} K W^{1/2}`
   - `L = chol(B)`
   - `a = b − W^{1/2} L^T\(L\(W^{1/2} K b))`
   - `f ← Ka`
3. Until `‖f_new − f‖ < tol`.

### Code — Newton Loop (`classification.py`, lines 121–171)
```python
n = X.shape[0]
f = np.zeros(n)   # Initialise at zero

for iteration in range(self.max_iter):
    # Gradient and Hessian of log-likelihood
    grad_log_lik = self._gradient_log_likelihood(f, y)   # ∇ log p(y|f)
    W = self._hessian_log_likelihood(f, y)               # diag of -∇∇ log p(y|f)

    # Build B = I + W^{1/2} K W^{1/2}
    W_sqrt    = np.sqrt(W)
    W_sqrt_K  = W_sqrt[:, np.newaxis] * self.K_          # W^{1/2} K
    B         = np.eye(n) + W_sqrt_K * W_sqrt[np.newaxis, :]

    L = stable_cholesky(B, jitter=self.noise)            # L = chol(B)

    b          = W * f + grad_log_lik                    # b = Wf + ∇log p
    W_sqrt_K_b = W_sqrt_K.dot(b)                         # W^{1/2} K b
    c          = np.linalg.solve(L, W_sqrt_K_b)
    c          = np.linalg.solve(L.T, c)                 # (L L^T)^{-1} W^{1/2} K b
    a          = b - W_sqrt * c                          # a = b - W^{1/2} L^T\(L\...)
    f_new      = self.K_.dot(a)                          # f ← Ka

    delta = np.max(np.abs(f_new - f))
    f = f_new
    if delta < self.tol:
        break
```

### 2.7.1 Gradient and Hessian of Log-Likelihood

| Likelihood | `∇ log p(y|f)` | `-∇∇ log p(y|f)` (W) |
|---|---|---|
| Logistic | `t − σ(f)`, where `t = (y+1)/2` | `σ(f)·(1−σ(f))` |
| Probit | `y·N(yf)·Φ(yf)^{-1}` | `z(z + yf)`, `z = N/Φ` |

**Code — Gradient** (`classification.py`, lines 296–313)
```python
def _gradient_log_likelihood(self, f, y):
    if self.likelihood == 'logistic':
        pi = self._sigmoid(f)
        t  = (y + 1.0) / 2.0      # {-1,+1} → {0,1}
        return t - pi              # residual in probability space
    else:  # probit
        yf = y * f
        norm_pdf = np.exp(-0.5*yf**2) / np.sqrt(2*np.pi)
        norm_cdf = self._probit_sigmoid(yf)
        return y * norm_pdf / norm_cdf
```

**Code — Hessian (W)** (`classification.py`, lines 315–339)
```python
def _hessian_log_likelihood(self, f, y):
    if self.likelihood == 'logistic':
        pi = self._sigmoid(f)
        W  = pi * (1.0 - pi)       # σ(f)(1-σ(f))
    else:  # probit
        yf = y * f
        z  = norm_pdf / norm_cdf
        W  = z * (z + yf)
    return np.maximum(W, 1e-10)    # ensure numerical positivity
```

---

## 2.8 GP Classification — Algorithm 3.2 (Predictions)

### Theory
Predictive probability at test point `x_*`:

```
f̄_* = k(X, x_*)^T · ∇ log p(y|f̂)
V[f_*] = k(x_*,x_*) − v^T v,   v = L^{-1}(W^{1/2} k(X, x_*))
π̄_* = ∫ σ(f_*) N(f_* | f̄_*, V[f_*]) df_*
```

The integral has no closed form for logistic — approximated using probit approximation.

### Code — `predict_proba` (`classification.py`, lines 180–231)
```python
def predict_proba(self, X):
    k_star      = self.kernel(self.X_train_, X)              # k(X, X_*)
    grad_log_lik = self._gradient_log_likelihood(self.f_hat_, self.y_train_)
    f_bar       = k_star.T.dot(grad_log_lik)                 # posterior mean of f_*

    W_sqrt      = np.sqrt(self.W_)
    v           = np.linalg.solve(self.L_, W_sqrt[:, np.newaxis] * k_star)
    k_star_star = self.kernel(X, X, diag=True)
    var_f       = np.maximum(k_star_star - np.sum(v**2, axis=0), 0.0)

    if self.likelihood == 'probit':
        kappa     = 1.0 / np.sqrt(1.0 + var_f)
        pi_star   = self._probit_sigmoid(kappa * f_bar)       # analytic
    else:  # logistic — use probit approximation
        pi_star   = self._logistic_averaged_probability(f_bar, var_f)

    proba = np.column_stack([1 - pi_star, pi_star])           # [P(0), P(1)]
    return proba
```

### Code — Logistic Averaged Probability Approximation (`classification.py`, lines 341–350)
```python
def _logistic_averaged_probability(self, f_bar, var_f):
    """σ(f) ≈ Φ(κf), κ² = π/8  →  integral ≈ Φ(κf̄ / √(1 + κ²V))"""
    kappa_sq     = np.pi / 8.0
    scaled_mean  = f_bar / np.sqrt(1.0 + kappa_sq * var_f)
    return self._probit_sigmoid(scaled_mean)
```

---

## 2.9 Likely Viva Questions

| Question | Short Answer | Where in Code |
|---|---|---|
| Why Cholesky and not direct inverse? | More stable, O(n³) but with smaller constant; avoids catastrophic cancellation. | `base.py` lines 15–79 |
| What is the log determinant computed from? | `log|K| = 2·Σ log(L_ii)` (diagonal of Cholesky). | `base.py` line 119 |
| Why optimise in log space? | Guarantees hyperparameters remain positive without explicit constraints. | `kernels.py` lines 111–118 |
| What does W represent in Alg. 3.1? | Diagonal of the negative Hessian — it's the local curvature of the log-likelihood. | `classification.py` lines 315–339 |
| Why are labels converted to {-1, +1}? | Standard convention in GPML for logistic/probit formulas. | `classification.py` line 112 |
| What is the Laplace approximation? | Replace intractable posterior with a Gaussian centred at the MAP estimate. | `classification.py` lines 81–171 |
| How is predictive variance computed? | `k(x_*,x_*) − v^T v`, subtracting the variance "explained" by the training data. | `regression.py` lines 187–190 |

---
---

# Assignment 3 — Variational Autoencoders (VAE & CVAE)

**Source files**: `Assignment3/part1_vae.py`, `Assignment3/part2_cvae.py`  
**Dataset**: MNIST (28×28 greyscale digits, 60 000 train / 10 000 test).

---

## 3.1 What Is a VAE?

A Variational Autoencoder is a **generative model** that learns a mapping between data space `x` and a structured latent space `z`. Unlike a plain autoencoder, the encoder outputs a distribution (`μ`, `σ²`) instead of a single point, forcing the latent space to be smooth and continuous.

The generative model is:
```
z ~ N(0, I)          (prior)
x | z ~ p_θ(x|z)     (decoder / likelihood)
```

The approximate posterior (encoder) is:
```
q_φ(z|x) = N(z | μ_φ(x), diag(σ²_φ(x)))
```

---

## 3.2 Architecture

### VAE (`part1_vae.py`, lines 109–192)

```
Input x (784)
  ↓ Linear(784 → 400) + ReLU          enc_hidden / enc_mu / enc_log_var
  ↓ Linear(400 → 20)  [two heads]     → μ  (Z_DIM=20)
                                       → log_var  (Z_DIM=20)
  ↓ Reparameterize: z = μ + σ·ε
  ↓ Linear(20 → 400) + ReLU           dec_hidden
  ↓ Linear(400 → 784)                 dec_out  (raw logits)
Output x̂_logits (784)
```

**Code — `__init__`** (`part1_vae.py`, lines 127–134)
```python
# Encoder
self.enc_hidden  = nn.Linear(input_dim, h_dim)    # 784 → 400
self.enc_mu      = nn.Linear(h_dim,     z_dim)    # 400 → 20
self.enc_log_var = nn.Linear(h_dim,     z_dim)    # 400 → 20  (two heads)
# Decoder
self.dec_hidden  = nn.Linear(z_dim,     h_dim)    # 20  → 400
self.dec_out     = nn.Linear(h_dim,     input_dim) # 400 → 784
```

### CVAE (`part2_cvae.py`, lines 142–161) — conditioning on label

```
Input: x (784) || one_hot(y) (10)  = 794 dimensions
  ↓ Linear(794 → 400) + ReLU
  ↓ → μ (20),  log_var (20)
  ↓ Reparameterize
  ↓ z (20) || one_hot(y) (10) = 30 dimensions → decoder
  ↓ Linear(30 → 400) + ReLU
  ↓ Linear(400 → 784)
```

**Code — CVAE `__init__`** (`part2_cvae.py`, lines 154–161)
```python
# Encoder takes x + label
self.enc_hidden  = nn.Linear(input_dim + num_classes, h_dim)  # 784+10=794 → 400
# Decoder takes z + label
self.dec_hidden  = nn.Linear(z_dim + num_classes, h_dim)      # 20+10=30 → 400
```

---

## 3.3 Encode Step

### Theory
The encoder maps input `x` to distribution parameters `(μ, log_var)` representing `q_φ(z|x)`.

Using `log_var` (instead of variance or std) is a **numerical stability** choice: it can take any real value (no clamping needed) and `σ = exp(0.5·log_var)` is always positive.

### Code — VAE `encode` (`part1_vae.py`, lines 136–149)
```python
def encode(self, x):
    h       = self.relu(self.enc_hidden(x))  # shared hidden representation
    mu      = self.enc_mu(h)                 # mean of q(z|x)
    log_var = self.enc_log_var(h)            # log variance of q(z|x)
    return mu, log_var
```

### Code — CVAE `encode` (`part2_cvae.py`, lines 163–179)
```python
def encode(self, x, y_onehot):
    inp = torch.cat([x, y_onehot], dim=1)   # (B, 794) — label concatenated
    h       = self.relu(self.enc_hidden(inp))
    mu      = self.enc_mu(h)
    log_var = self.enc_log_var(h)
    return mu, log_var
```

---

## 3.4 Reparameterization Trick

### Theory
We need to sample `z ~ q_φ(z|x) = N(μ, σ²)` and **backpropagate gradients through the sample**. Direct sampling is non-differentiable. The reparameterization trick rewrites:

```
z = μ + σ · ε,     ε ~ N(0, I)
```

Now `z` is a deterministic function of `μ`, `σ`, and the **fixed noise** `ε`. Gradients flow through `μ` and `σ`.

### Code (`part1_vae.py`, lines 151–164; identical in CVAE)
```python
@staticmethod
def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std     = torch.exp(0.5 * log_var)    # σ = exp(log_var / 2)
    epsilon = torch.randn_like(std)       # ε ~ N(0, I), same shape as std
    return mu + std * epsilon             # z = μ + σ·ε  (differentiable w.r.t. μ, σ)
```
`torch.randn_like` creates `ε` on the same device/dtype as `std` without any graph connection, so gradients propagate only through `mu` and `std`.

---

## 3.5 Decode Step

### Theory
The decoder takes `z` (and optionally the label for CVAE) and maps it back to reconstruction logits. The sigmoid is applied **outside** the model (not in `decode`) for two reasons:
1. `BCEWithLogitsLoss` is numerically more stable than `BCELoss(sigmoid(logits))`.
2. During sampling/generation we can still apply sigmoid explicitly.

### Code — VAE `decode` (`part1_vae.py`, lines 166–176)
```python
def decode(self, z):
    h = self.relu(self.dec_hidden(z))
    return self.dec_out(h)               # returns raw logits (pre-sigmoid)
```

### Code — CVAE `decode` (`part2_cvae.py`, lines 196–209)
```python
def decode(self, z, y_onehot):
    inp = torch.cat([z, y_onehot], dim=1)   # (B, 30) — z || label
    h   = self.relu(self.dec_hidden(inp))
    return self.dec_out(h)
```

---

## 3.6 ELBO Loss Function

### Theory
The VAE objective is the **Evidence Lower BOund (ELBO)**:

```
ELBO = E_{q(z|x)}[log p(x|z)]  −  KL(q(z|x) || p(z))
     = Reconstruction term      −  Regularisation term
```

We **maximise** ELBO (equivalently, **minimise** negative ELBO = Loss).

**Reconstruction term**: Binary Cross-Entropy between `x` and `sigmoid(x̂_logits)`:

```
Recon = −Σ_i [x_i log σ(x̂_i) + (1−x_i) log(1−σ(x̂_i))]
```

**KL divergence** (closed form for Gaussian vs. standard Gaussian):

```
KL = −½ Σ_j (1 + log σ²_j − μ²_j − σ²_j)
   = −½ Σ_j (1 + log_var_j − mu_j² − exp(log_var_j))
```

### Code — `vae_loss` (`part1_vae.py`, lines 202–224)
```python
bce_logits_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

def vae_loss(x, x_hat_logits, mu, log_var, beta=1.0):
    batch_size = x.size(0)

    # Reconstruction: fused sigmoid + BCE (numerically stable under AMP)
    recon = bce_logits_loss_fn(x_hat_logits, x) / batch_size

    # KL: closed-form for N(μ,σ²) vs N(0,I)
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size

    total = recon + beta * kl    # β controls KL weight (annealing)
    return total, recon, kl
```
Identical formula in CVAE (`cvae_loss`, `part2_cvae.py` lines 237–259).

---

## 3.7 KL Annealing (β-VAE Schedule)

### Theory
**Posterior collapse**: early in training, the decoder is too weak to use `z`, so the encoder collapses to the prior (`μ ≈ 0`, `σ ≈ 1`) and the KL term vanishes. To prevent this, the KL weight `β` is linearly ramped from 0 to 1 over the first `KL_ANNEAL_EPOCHS` epochs.

During warm-up (`β ≈ 0`), the model is essentially a standard autoencoder — it learns to encode information. Once `β = 1`, the full VAE objective is applied.

### Code (`part1_vae.py`, lines 249–251 in training loop)
```python
# KL annealing: linearly ramp β from 0 → 1 over first KL_ANNEAL_EPOCHS
beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)   # epoch=1: β=0.1, ..., epoch=10: β=1.0
```
`KL_ANNEAL_EPOCHS = 10` (`part1_vae.py` line 61), so full ELBO is active from epoch 10 onward.

---

## 3.8 Training Loop with Mixed-Precision (AMP)

### Theory
NVIDIA's Automatic Mixed Precision (AMP) uses `float16` where safe (large matrix ops) and `float32` where needed (accumulated sums, loss). A `GradScaler` scales the loss before `.backward()` to prevent underflow in `float16` gradients.

### Code (`part1_vae.py`, lines 255–283)
```python
for epoch in range(1, NUM_EPOCHS + 1):
    beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)   # KL annealing

    for i, (x, _) in enumerate(loop):
        x = x.view(-1, INPUT_DIM).to(device, non_blocking=True)

        # AMP forward pass in float16
        with torch.amp.autocast("cuda", dtype=torch.float16):
            x_hat, mu, log_var = model(x)
            loss, recon, kl    = vae_loss(x, x_hat, mu, log_var, beta)

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()
        scaler.scale(loss).backward()          # scale loss for float16 stability
        scaler.step(optimizer)                 # unscale, check for inf/nan, then step
        scaler.update()                        # update scaler factor
```
For CVAE, the same structure is used but `model(x, y_onehot)` passes the label (`part2_cvae.py` line 298).

---

## 3.9 Early Stopping

### Theory
After the KL annealing phase, the best model is tracked and training stops if no improvement for `EARLY_STOP_PATIENCE` consecutive epochs.

### Code (`part1_vae.py`, lines 306–322)
```python
if epoch > KL_ANNEAL_EPOCHS:                 # only after β=1
    if avg_total < best_loss:
        best_loss       = avg_total
        patience_counter = 0
        torch.save(model.state_dict(), ...best_model.pt...)
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"[EARLY STOP] ...")
            break
else:
    # During annealing: always update best model (β changing so loss not comparable)
    best_loss = avg_total
    torch.save(model.state_dict(), ...best_model.pt...)
```

---

## 3.10 Generation (Sampling from Prior)

### Theory
After training, to generate new images: sample `z ~ N(0, I)`, then decode to get `x̂`.

### Code — VAE Sampling (`part1_vae.py`, lines 402–405)
```python
z       = torch.randn(64, Z_DIM, device=device)          # sample from prior
samples = torch.sigmoid(model.decode(z)).view(-1, 1, 28, 28)  # decode + sigmoid
```

### Code — CVAE Conditional Generation (`part2_cvae.py`, lines 442–449)
```python
for digit in range(NUM_CLASSES):
    z      = torch.randn(num_samples_per_digit, Z_DIM, device=device)
    labels = torch.full((num_samples_per_digit,), digit, ...)
    y_onehot = one_hot(labels, device=device)
    samples  = torch.sigmoid(model.decode(z, y_onehot)).view(-1, 1, 28, 28)
```
By fixing the label `y` while sampling `z ~ N(0,I)`, the CVAE generates images of a specific digit on demand.

---

## 3.11 Latent Interpolation

### Theory
Linearly interpolate between two latent codes `μ_a` and `μ_b`:
```
z_α = (1 − α)·μ_a + α·μ_b,   α ∈ [0, 1]
```
Decoding each `z_α` gives a morphing sequence between two images.

### Code — VAE Interpolation (`part1_vae.py`, lines 490–499)
```python
with torch.no_grad():
    mu_a, _ = model.encode(img_a)
    mu_b, _ = model.encode(img_b)

for alpha in np.linspace(0, 1, num_steps):
    z_interp = (1 - alpha) * mu_a + alpha * mu_b
    decoded  = torch.sigmoid(model.decode(z_interp)).view(1, 1, 28, 28)
```

### Code — CVAE Interpolation (`part2_cvae.py`, lines 503–513)
```python
for step, alpha in enumerate(np.linspace(0, 1, num_steps)):
    z_interp = (1 - alpha) * mu_1 + alpha * mu_7
    y_interp = (1 - alpha) * y_oh_1 + alpha * y_oh_7  # also interpolate the label!
    decoded  = torch.sigmoid(model.decode(z_interp, y_interp))
```
The CVAE additionally interpolates the **one-hot label vector**, creating a smooth transition in both latent and class-conditioning spaces.

---

## 3.12 One-Hot Encoding Helper

**Code** (`part2_cvae.py`, lines 111–125)
```python
def one_hot(labels: torch.Tensor, num_classes: int = NUM_CLASSES,
            device: str = "cuda") -> torch.Tensor:
    """
    Creates a (B, 10) tensor where position y is set to 1.
    Uses scatter_ for efficiency — no Python loop over batch.
    """
    return torch.zeros(labels.size(0), num_classes, device=device).scatter_(
        1, labels.unsqueeze(1), 1   # scatter 1 at (batch_idx, label_idx)
    )
```

---

## 3.13 VAE vs CVAE — Key Differences

| Aspect | VAE | CVAE |
|---|---|---|
| Encoder input | `x` (784) | `x ∥ one_hot(y)` (794) |
| Decoder input | `z` (20) | `z ∥ one_hot(y)` (30) |
| Generation | Uncontrolled — random digit | Controlled — specific digit |
| Interpolation | Interpolate `z` only | Interpolate `z` **and** `y` |
| `enc_hidden` size | `Linear(784, 400)` | `Linear(794, 400)` |
| `dec_hidden` size | `Linear(20, 400)` | `Linear(30, 400)` |

---

## 3.14 Likely Viva Questions

| Question | Short Answer | Where in Code |
|---|---|---|
| Why is reparameterization needed? | Sampling breaks the gradient flow; re-parameterizing as `μ + σ·ε` makes the sample differentiable. | `part1_vae.py` lines 151–164 |
| Why use `log_var` instead of `var`? | Log can be any real value; variance must be positive. Avoids clamping. | `part1_vae.py` line 148 |
| What is the KL divergence formula? | `−½ Σ(1 + log_var − μ² − exp(log_var))`. | `part1_vae.py` line 222 |
| What is posterior collapse? | Encoder collapses to prior — KL = 0, decoder ignores z. KL annealing prevents this. | `part1_vae.py` lines 249–251 |
| Why `BCEWithLogitsLoss` and not `BCELoss`? | Fused sigmoid + log is numerically stable and AMP-safe. | `part1_vae.py` lines 198–221 |
| How does CVAE differ from VAE structurally? | Label concatenated to both encoder input and decoder input. | `part2_cvae.py` lines 154–161 |
| How is generation controlled in CVAE? | Sample `z ~ N(0,I)` and provide desired `y_onehot` to `decode(z, y_onehot)`. | `part2_cvae.py` lines 442–449 |
| What does the KL term in ELBO do? | Regularises the latent space towards `N(0,I)`, ensuring a continuous, sampleable space. | `part1_vae.py` lines 222–223 |
| Why is early stopping only applied after KL annealing? | During annealing β changes, so loss values are not comparable across epochs. | `part1_vae.py` lines 306–322 |
| What does `GradScaler` do? | Scales loss before backprop to avoid float16 underflow; unscales before optimizer step. | `part1_vae.py` lines 266–268 |

---

# Quick Concept-to-File Index

| Concept | File | Lines |
|---|---|---|
| GBT Regression initialisation (`F_0 = mean(y)`) | `Assignment1/src/gbt/core.py` | 127–129 |
| GBT Classification initialisation (log-odds) | `Assignment1/src/gbt/core.py` | 273–279 |
| MSE pseudo-residuals | `Assignment1/src/gbt/utils.py` | 25–32 |
| Logistic pseudo-residuals | `Assignment1/src/gbt/utils.py` | 61–74 |
| MSE leaf optimisation (mean) | `Assignment1/src/gbt/utils.py` | 35–43 |
| Logistic leaf optimisation (Newton step) | `Assignment1/src/gbt/utils.py` | 77–109 |
| Leaf-to-gamma map construction | `Assignment1/src/gbt/core.py` | 160–167 |
| Model update with shrinkage | `Assignment1/src/gbt/core.py` | 169–172 |
| Row subsampling | `Assignment1/src/gbt/core.py` | 79–86 |
| Prediction accumulation | `Assignment1/src/gbt/core.py` | 199–222 |
| Numerically stable Cholesky | `Assignment2/src/gp/base.py` | 39–54 |
| Cholesky solve | `Assignment2/src/gp/base.py` | 57–79 |
| Log marginal likelihood | `Assignment2/src/gp/base.py` | 82–124 |
| RBF kernel | `Assignment2/src/gp/kernels.py` | 68–137 |
| Matérn kernel | `Assignment2/src/gp/kernels.py` | 140–232 |
| Rational Quadratic kernel | `Assignment2/src/gp/kernels.py` | 235–316 |
| GP Regression fit (Alg. 2.1 steps 1–3) | `Assignment2/src/gp/regression.py` | 117–131 |
| GP Regression predict (Alg. 2.1 steps 4–5) | `Assignment2/src/gp/regression.py` | 133–191 |
| Hyperparameter optimisation (L-BFGS-B) | `Assignment2/src/gp/regression.py` | 236–302 |
| GP Classification Newton loop (Alg. 3.1) | `Assignment2/src/gp/classification.py` | 121–171 |
| GP Classification predictions (Alg. 3.2) | `Assignment2/src/gp/classification.py` | 180–231 |
| Logistic gradient & Hessian | `Assignment2/src/gp/classification.py` | 296–339 |
| VAE architecture | `Assignment3/part1_vae.py` | 109–192 |
| CVAE architecture | `Assignment3/part2_cvae.py` | 131–228 |
| Reparameterization trick | `Assignment3/part1_vae.py` | 151–164 |
| ELBO loss (VAE) | `Assignment3/part1_vae.py` | 202–224 |
| ELBO loss (CVAE) | `Assignment3/part2_cvae.py` | 237–259 |
| KL annealing | `Assignment3/part1_vae.py` | 249–251 |
| AMP training loop | `Assignment3/part1_vae.py` | 255–283 |
| Early stopping | `Assignment3/part1_vae.py` | 306–322 |
| Prior sampling (generation) | `Assignment3/part1_vae.py` | 402–405 |
| Conditional generation (CVAE) | `Assignment3/part2_cvae.py` | 442–449 |
| Latent interpolation (VAE) | `Assignment3/part1_vae.py` | 490–499 |
| Latent interpolation + label (CVAE) | `Assignment3/part2_cvae.py` | 503–513 |
