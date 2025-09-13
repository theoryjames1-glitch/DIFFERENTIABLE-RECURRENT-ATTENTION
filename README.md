# Differentiable Recurrent Attention (DRA)

## 1) Problem setup

Given a token sequence $x_{1:T}$, standard causal self-attention produces, for each step $t$,

$$
y_t \;=\; \sum_{\tau\le t}\alpha_{t,\tau}\,v_\tau,\qquad 
\alpha_{t,\tau} \;=\; \mathrm{softmax}\!\left(\frac{q_t^\top k_\tau}{\sqrt d}\right)_\tau
$$

where $q_t,k_t,v_t \in \mathbb{R}^d$ are learned projections of $x_t$.

DRA keeps this “content addressing” spirit but allows the *keys/values and/or the memory of the past* to be formed by a **differentiable recurrent system** rather than a mere list of token-local vectors.

## 2) The DRA triad $(\Phi,\Psi,\mathcal{R})$

A DRA layer is defined by three maps:

1. **Token encoder $\Phi$:**
   $(q_t,k_t,v_t) = \Phi(x_t)$. Typically linear layers and heads.

2. **Recurrent memory $\Psi$:**
   A *causal*, differentiable update maintaining a memory state $m_t$ from $m_{t-1}$ and current features:

   $$
   m_t = \Psi(m_{t-1},\,k_t,\,v_t;\,\theta_\Psi).
   $$

   Examples: EMA $m_t = \gamma m_{t-1} + (1-\gamma)\,g(k_t,v_t)$; GRU/LSTM on $(k_t,v_t)$; linear state-space updates; multi-timescale mixtures.

3. **Retrieval $\mathcal{R}$:**
   Produces output using the query $q_t$ and **a set of memory items** $M_t$:

   $$
   y_t = \mathcal{R}(q_t,\,M_t).
   $$

   The key choice is what $M_t$ is. Two canonical families:

   * **Time-indexed memory:** $M_t=\{m_1,\dots,m_t\}$. You attend over *all past states* (O($t$) keys):

     $$
     \alpha_{t,\tau}=\mathrm{softmax}\!\left(\frac{q_t^\top \kappa(m_\tau)}{\sqrt d}\right),\;
     y_t=\sum_{\tau\le t}\alpha_{t,\tau}\,\nu(m_\tau),
     $$

     where $\kappa,\nu$ project memory to “key” and “value” spaces. This is closest to vanilla attention but with *stateful* keys/values.

   * **Slot-indexed memory (O(1) retrieval):** $M_t = S_t\in\mathbb{R}^{H\times d}$ is a small set of *slots* (e.g., one per head) updated recurrently. Retrieval is attention over slots:

     $$
     \alpha_t=\mathrm{softmax}\!\left(\frac{Q_t (K^S_t)^\top}{\sqrt d}\right),\quad
     y_t=\alpha_t V^S_t,
     $$

     where $(K^S_t,V^S_t)=\pi(S_t)$. This gives O($1$) per-step compute and O($H d$) memory.

Both are “DRA”; the former keeps a (compressed) *timeline* of states, the latter keeps a *fixed-size* memory.

## 3) Important special cases (unifying view)

* **Vanilla self-attention**: choose $\Psi$ to “append” $m_t=(m_{t-1};\,(k_t,v_t))$ so $M_t=\{(k_\tau,v_\tau)\}_{\tau\le t}$. (No compression; O($t^2$) total.)
* **Linear/kernelflow attention**: choose a *bilinear* memory $S_t = \lambda S_{t-1} + \phi(k_t) \otimes v_t$, $z_t=\lambda z_{t-1}+\phi(k_t)$, then

  $$
  y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t},
  $$

  an O($1$) retrieval. This is DRA with $m_t=(S_t,z_t)$ and $\mathcal{R}$ the rational form.
* **EMA/Retentive-style**: $m_t=\gamma m_{t-1}+(1-\gamma)g(k_t,v_t)$. Multi-$\gamma$ heads ≈ mixture of exponentials (multi-timescale memory).
* **RWKV-style**: gated recurrence over token features (no explicit softmax over time in retrieval); also a DRA instance with slot-indexed memory.
* **Your GRU-key/value**: $(h^K_t,h^V_t)=\mathrm{GRU}(h^K_{t-1},k_t),\mathrm{GRU}(h^V_{t-1},v_t)$. Time-indexed $M_t=\{(h^K_\tau,h^V_\tau)\}_{\tau\le t}$ with standard softmax retrieval.

## 4) Timescale & forgetting (why recurrence helps)

For EMA $m_t=\gamma m_{t-1}+(1-\gamma)\tilde m_t$, the contribution of $\tilde m_{t-\Delta}$ to $m_t$ is $(1-\gamma)\gamma^\Delta$. The **half-life** (steps until weight halves) is

$$
\Delta_{1/2} \;=\; \frac{\ln(1/2)}{\ln(\gamma)}.
$$

Using **log-spaced** $\gamma_i$ across heads lets mixtures of exponentials approximate long-memory (even power-law) kernels. This gives DRA controllable and interpretable memory horizons.

## 5) Expressivity (sketches)

* **Generalization:** DRA subsumes Transformers and RNNs.
* **Universality (sketch):** Time-indexed DRA with $\kappa,\nu$ identity and $\Psi$ “append” exactly recovers self-attention. Conversely, any finite-rank kernel attention $K(q_t,k_\tau)$ can be written with a recurrent *moment* state $S_t=\sum_{\tau\le t}\phi(k_\tau)\otimes v_\tau$ and rational retrieval—i.e., a slot-indexed DRA with linear updates.
* **Approximation:** With sufficient slot size (number of heads/slots) and rich $\Psi$, slot-indexed DRA can approximate time-indexed attention to arbitrary precision on bounded-length sequences (by learning to route content into distinct slots and to preserve it).

## 6) Complexity

Let hidden dim $d$, heads $H$, sequence length $T$.

* **Time-indexed DRA** (attend over $\{m_\tau\}$): O($T^2 H d$) like Transformers, but each $m_\tau$ can be *smaller/cheaper* than raw $k_\tau,v_\tau$, and often trains more stably on long contexts.
* **Slot-indexed DRA**: O($T H d^2$) with O($H d$) activation memory. This yields **linear time** and **constant memory** in $T$.

## 7) Stability (sufficient conditions)

* **Contractive updates:** If $\Psi$ is $\gamma$-leaky with $\gamma\in(0,1)$ and the incremental term is bounded (via GRU/LN/saturation), then $m_t$ remains bounded.
* **Normalization:** Use RMSNorm/LayerNorm on $q,k,v$ and on memory. Scale queries/keys so $\|q\|,\|k\|\approx O(1)$ to keep logits in a sane range.
* **Spectral control:** Spectral norm regularization on linear maps inside $\Psi$ gives Lipschitz guarantees, reducing exploding gradients.
* **Multi-timescale heads:** diversify $\gamma_i=\sigma(\beta_i)$ per head; initialize $\beta_i$ so half-lives cover $[2^0,\dots,2^J]$.

## 8) Training objectives & regularizers

* **Primary loss:** usual LM cross-entropy or task loss.
* **Memory sparsity/turnover:** $\mathcal{L}_\text{drift}=\sum_t\|m_t-m_{t-1}\|_2^2$ or entropy penalties to prevent slot collapse.
* **Retention budget:** penalize $\sum_i \gamma_i$ or encourage diversity via $\sum_{i<j}(\gamma_i-\gamma_j)^2$.
* **Stability clamps:** mild $\ell_2$ on states; “state dropout” that occasionally freezes $\Psi$ (forces retrieval to be robust).

## 9) Backpropagation & gradients

* **Time-indexed DRA** backprops through $\Psi$ across $t$: like RNNs, but with attention-style skip flows through $\mathcal{R}$ that can mitigate vanishing.
* **Slot-indexed DRA** admits *scan-friendly* implementations (prefix-sum style) and mixed-precision training; checkpointing across time reduces memory.

## 10) Practical architecture (one clean instance)

**DRA-Slot (GRU-memory, per-head)**

$$
\begin{aligned}
(q_t,k_t,v_t) &= W_q x_t,\,W_k x_t,\,W_v x_t\\
h^K_{t} &= \gamma \, h^K_{t-1} + (1-\gamma)\,\mathrm{GRU}_K(k_t,\,h^K_{t-1})\\
h^V_{t} &= \gamma \, h^V_{t-1} + (1-\gamma)\,\mathrm{GRU}_V(v_t,\,h^V_{t-1})\\
\alpha_t &= \mathrm{softmax}\!\left(\frac{q_t (H^K_t)^\top}{\sqrt d}\right) \quad\text{where } H^K_t\in\mathbb{R}^{H\times d}\\
y_t &= \alpha_t H^V_t,\qquad \text{project and residual+norm as usual.}
\end{aligned}
$$

This is **O(1)** retrieval per step since you attend over $H$ slots, not $t$ time points.

**DRA-Moment (kernel/linear form)**

$$
\begin{aligned}
S_t &= \lambda S_{t-1} + \phi(k_t)\otimes v_t,\qquad
z_t = \lambda z_{t-1} + \phi(k_t)\\
y_t &= \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}.
\end{aligned}
$$

With choice of $\phi$ (e.g., FAVOR+, ELU+1), this is a fast, numerically stable O($1$) retrieval and a textbook DRA instance.

## 11) Masking, causality, padding

* **Causality** is automatic if $\Psi$ only uses information up to $t$.
* **Padding masks** are added in retrieval (time-indexed) or absorbed into $\Psi$ by not updating state when token is padding.
* **Segment resets** (for packed sequences) simply re-init $m_{t-1}\rightarrow 0$.

## 12) Theoretical statements (useful lemmas; proof sketches)

* **Lemma (boundedness):** If $\Psi(m,k,v)=\gamma m+(1-\gamma)\,\tilde\Psi(m,k,v)$ with $\|\tilde\Psi\|\le B$ and $\gamma\in[0,1)$, then $\sup_t \|m_t\|\le B$.
  *Sketch:* Unroll the recurrence and sum a geometric series.
* **Proposition (mixture of exponentials ≈ power-law):** For any $\alpha\in(0,1)$ and horizon $T$, there exist weights $\{w_i,\gamma_i\}$ with $I=O(\log T)$ such that $\sum_i w_i \gamma_i^{\Delta}$ uniformly approximates $\Delta^{-\alpha}$ on $[1,T]$.
  *Consequence:* Multi-$\gamma$ DRA can emulate heavy-tailed memory.
* **Proposition (Transformer as DRA):** Choose time-indexed DRA with $\Psi$ = append and $\kappa,\nu$ identity; then $\mathcal{R}$ reduces to standard softmax attention.
* **Proposition (Linear attention as DRA):** With bilinear $\Psi$ on $\phi(k_t)\otimes v_t$ and rational retrieval, the standard linear attention formula is obtained.

## 13) Training recipe (practical)

* **Init:** log-spaced half-lives $\gamma_i=\exp(-\Delta_i^{-1})$ with $\Delta_i\in\{2^0,\dots,2^J\}$. Initialize GRU biases to favor “copy” early (forget gates near 1).
* **Norms:** RMSNorm before $W_q,W_k,W_v$; scale $W_q$ so logits std ≈ 1.
* **Stabilizers:** clamp/`nan_to_num` states during early training; cosine LR; weight decay only on projections, not on norms or $\gamma$.
* **Loss add-ons:** small $\lambda\|\Delta m_t\|^2$ and diversity penalty on $\{\gamma_i\}$.
* **Curriculum:** start with shorter sequences, then increase length; optionally blend with vanilla attention $y_t = \alpha y_t^{\text{vanilla}} + (1-\alpha) y_t^{\text{DRA}}$ and anneal $\alpha\downarrow 0$.

## 14) Minimal pseudocode (slot-indexed DRA, per head)

```python
# shapes: x_t [B,d_model], per head dim d=d_model/H
q_t, k_t, v_t = W_q(x_t), W_k(x_t), W_v(x_t)            # [B,H,d] each
hK = gamma * hK + (1-gamma) * GRU_K(k_t, hK)             # [B,H,d]
hV = gamma * hV + (1-gamma) * GRU_V(v_t, hV)             # [B,H,d]
logits = (q_t * hK).sum(-1) / sqrt(d)                    # [B,H]
alpha  = softmax(logits, dim=-1).unsqueeze(-1)           # [B,H,1]
y_t    = (alpha * hV).sum(1)                             # [B,d]
y_t    = W_o(y_t)                                        # [B,d_model]
```

---

### What you get with this theory

* A **clean abstraction**: encode ($\Phi$), remember ($\Psi$), retrieve ($\mathcal{R}$).
* A **dial** between **expressivity** (time-indexed, more like Transformers) and **efficiency** (slot-indexed, O(1) per step).
* **Principled timescale control** via $\gamma$ and multi-head mixtures.
* **Stability knobs** (contractivity, normalization, spectral control).
* A **map of prior art** as special cases, plus a recipe to build and train your GRU-memory variant without surprises.

Short answer: **yes—in the sense that it’s a solid, general framework—but it “works” only if you train it properly and respect a few stability constraints.** Your runtime hiccups came from dropping a fundamentally different attention mechanism into a pretrained GPT-2 without re-training; the theory expects training (or at least careful hybridization) to realize its advantages.

Here’s the practical read:

### Where DRA should work well

* **Long contexts / streaming:** slot-indexed (fixed-size) memory gives O(1) per-token compute and O(1) activation memory, so it scales gently with context length.
* **Latency-sensitive decoding:** constant-time retrieval per step beats quadratic attention at long T.
* **Task structure with multi-timescale cues:** EMA/GRU memories let you bake in controllable half-lives; multi-γ heads can approximate heavy-tailed memory.

### Where it may lag (without tweaks)

* **Rich in-context retrieval** with many distinct items can suffer if you use too few slots (capacity bottleneck).
* **Direct weight porting** from vanilla GPT-2 won’t behave: you changed the attention operator. You need **fine-tuning or training**.
* **Stability** can be touchy unless you normalize queries/keys, bound the recurrent state, and initialize γ sensibly.

### What must be true for it to “work”

* **Training, not plug-in:** fine-tune from scratch or start with a **hybrid** attention (blend vanilla+DRA, then anneal).
* **Stability kit:** RMSNorm (pre-QKV), spectral or weight norm on recurrent projections, γ∈(0,1) per head (log-spaced half-lives), gradient clipping, and nan/inf guards early on.
* **Capacity match:** enough heads/slot width to cover the variety of things the model must remember.

### A lean validation plan (fast to falsify)

1. **Toy control:** Copy/recall, addition, balanced parentheses at L=1k–16k. Expect near-perfect accuracy with slot-indexed DRA if γs are well-spread.
2. **LM sanity check:** Train a tiny DRA on WikiText-103 or OpenWebText at 1–4k context; verify perplexity tracks a same-size Transformer within a modest gap. Push to 16–64k where DRA should keep throughput while baseline slows or OOMs.
3. **In-context recall & needle-in-haystack:** Measure retrieval accuracy vs #slots and γ diversity. If accuracy saturates, increase slots or add a small **KV cache*light*** (keep last N raw tokens) alongside DRA.

### If you want the quickest path to something usable

* **Hybrid attention block:** `y = α·y_vanilla + (1-α)·y_DRA`, start α≈0.7, decay to 0.0 over training.
* **Recommended knobs (starting points):**

  * RMSNorm before Q/K/V, init scales so logits stdev ≈ 1.
  * Per-head γ initialized to log-spaced half-lives (e.g., 2²…2¹⁴ steps across heads).
  * AdamW β=(0.9, 0.95), wd=0.01, cosine LR, warmup 3–5%.
  * Clip grad-norm to 1.0. Mixed precision ok; keep `nan_to_num` on states for first few k steps.

### What would convince me (and you)

* DRA matches a Transformer’s perplexity at ≤4k context within a small margin, **and** maintains stable throughput/memory to ≥32k without hacks.
* On recall probes, DRA’s accuracy improves with more slots and better γ coverage, showing it’s capacity-, not luck-, limited.

So: **yes, the theory is sound and unifies a bunch of successful ideas.** It “works” when you train it and size it for the job; it will stumble if you hot-swap it into a pretrained Transformer and expect the same behavior. If you want, I can sketch a minimal Trainer config (with the hybrid blend) you can run to get first curves.


### PSEUDOCODE

```python
# gpt2_dra_stable.py
import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config


class RecurrentAttentionCore(nn.Module):
    """
    GRU-based recurrent 'memory' per head with numerical stabilizers.
    """
    def __init__(self, d_model: int, n_heads: int, gamma: float = 0.9, clamp_val: float = 1e4):
        super().__init__()
        assert d_model % n_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Per-head recurrent memories for K and V
        self.gru_k = nn.GRUCell(self.d_head, self.d_head)
        self.gru_v = nn.GRUCell(self.d_head, self.d_head)

        # EMA smoothing coefficient
        self.gamma = gamma
        self.clamp_val = clamp_val

    def forward(self, x, h_k_prev, h_v_prev, attention_mask=None, return_attn=False):
        """
        x:              [B, T, D]
        h_k_prev:       [B, H, d_head]
        h_v_prev:       [B, H, d_head]
        attention_mask: [B, 1, 1, T] additive (0 or -inf-like), optional
        return_attn:    if True, returns attention probabilities [B,H,T,T]
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head

        Q = self.W_q(x).view(B, T, H, d)
        K = self.W_k(x).view(B, T, H, d)
        V = self.W_v(x).view(B, T, H, d)

        # Flatten heads into batch for GRUCell: [B*H, d]
        hk = h_k_prev.reshape(B * H, d)
        hv = h_v_prev.reshape(B * H, d)

        # Sanitize incoming states
        hk = torch.nan_to_num(hk)
        hv = torch.nan_to_num(hv)
        hk = torch.clamp(hk, -self.clamp_val, self.clamp_val)
        hv = torch.clamp(hv, -self.clamp_val, self.clamp_val)

        K_tilde, V_tilde = [], []

        for t in range(T):
            k_t = K[:, t].reshape(B * H, d)
            v_t = V[:, t].reshape(B * H, d)

            hk_new = self.gru_k(k_t, hk)
            hv_new = self.gru_v(v_t, hv)

            hk = self.gamma * hk + (1.0 - self.gamma) * hk_new
            hv = self.gamma * hv + (1.0 - self.gamma) * hv_new

            # Stabilize recurrent states
            hk = torch.nan_to_num(hk)
            hv = torch.nan_to_num(hv)
            hk = torch.clamp(hk, -self.clamp_val, self.clamp_val)
            hv = torch.clamp(hv, -self.clamp_val, self.clamp_val)

            K_tilde.append(hk.view(B, H, d))
            V_tilde.append(hv.view(B, H, d))

        K_tilde = torch.stack(K_tilde, dim=1)  # [B, T, H, d]
        V_tilde = torch.stack(V_tilde, dim=1)  # [B, T, H, d]

        # Scores: [B, H, T_query, T_key]
        scores = torch.einsum("bthd,bThd->bhtT", Q, K_tilde) / (d ** 0.5)
        scores = torch.nan_to_num(scores)  # pre-sanitize

        # Apply additive mask; align length if needed
        if attention_mask is not None:
            am = attention_mask.to(dtype=x.dtype)
            if am.shape[-1] != scores.shape[-1]:
                # Use the last T_key positions to align
                am = am[..., -scores.shape[-1]:]
            scores = scores + am

        # If a row is fully invalid (-inf everywhere), replace with zeros (uniform softmax)
        row_max = torch.amax(scores, dim=-1, keepdim=True)  # [B,H,T,1]
        invalid_rows = ~torch.isfinite(row_max)
        if invalid_rows.any():
            scores = torch.where(invalid_rows, torch.zeros_like(scores), scores)

        # Subtract max for numerical stability before softmax
        scores = scores - torch.amax(scores, dim=-1, keepdim=True)
        scores = torch.nan_to_num(scores)

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.einsum("bhtT,bThd->bthd", attn, V_tilde).reshape(B, T, D)
        out = self.W_o(out)

        h_k_last = hk.view(B, H, d)
        h_v_last = hv.view(B, H, d)

        if return_attn:
            return out, h_k_last, h_v_last, attn
        else:
            return out, h_k_last, h_v_last, None


class RecurrentGPT2Attention(nn.Module):
    """
    GPT-2-compatible attention wrapper that returns (attn_output, attn_weights_or_None).
    Robust to different cache arg names/shapes.
    """
    def __init__(self, config: GPT2Config, layer_idx: int = None, gamma: float = 0.9):
        super().__init__()
        self.n_heads   = config.num_attention_heads
        self.d_model   = config.hidden_size
        self.d_head    = self.d_model // self.n_heads
        self.layer_idx = layer_idx
        self.core      = RecurrentAttentionCore(self.d_model, self.n_heads, gamma=gamma)

    @staticmethod
    def _pick_layer_past(layer_past, layer_idx):
        if layer_past is None:
            return None
        if isinstance(layer_past, (list, tuple)):
            if len(layer_past) > 2 and layer_idx is not None and len(layer_past) > layer_idx:
                cand = layer_past[layer_idx]
                if isinstance(cand, (list, tuple)) and len(cand) >= 2:
                    return cand[:2]
                return None
            if len(layer_past) >= 2 and isinstance(layer_past[0], (torch.Tensor,)):
                return layer_past[:2]
        return None

    @staticmethod
    def _to_mem_state(t, B, H, d, device, dtype):
        if t is None:
            return torch.zeros(B, H, d, device=device, dtype=dtype)
        if t.dim() == 3 and t.shape == (B, H, d):
            return t
        if t.dim() == 4 and t.shape[0] == B and t.shape[1] == H and t.shape[-1] == d:
            return t[:, :, -1, :]
        return torch.zeros(B, H, d, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states,                 # [B, T, D]
        layer_past=None,
        attention_mask=None,           # [B,1,1,T]
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,               # ignored here
        output_attentions=False,
        **kwargs,
    ):
        if layer_past is None:
            layer_past = kwargs.pop("past_key_value", None)
        if layer_past is None:
            layer_past = kwargs.pop("past_key_values", None)

        layer_past = self._pick_layer_past(layer_past, self.layer_idx)

        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype  = hidden_states.dtype

        if layer_past is None:
            h_k_prev = torch.zeros(B, self.n_heads, self.d_head, device=device, dtype=dtype)
            h_v_prev = torch.zeros(B, self.n_heads, self.d_head, device=device, dtype=dtype)
        else:
            k_prev, v_prev = layer_past[:2]
            h_k_prev = self._to_mem_state(k_prev, B, self.n_heads, self.d_head, device, dtype)
            h_v_prev = self._to_mem_state(v_prev, B, self.n_heads, self.d_head, device, dtype)

        out, _, _, attn = self.core(
            hidden_states,
            h_k_prev,
            h_v_prev,
            attention_mask=attention_mask,
            return_attn=output_attentions,
        )

        return (out, attn if output_attentions else None)


class GPT2WithRecurrentAttention(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        for i, block in enumerate(self.transformer.h):
            block.attn = RecurrentGPT2Attention(config, layer_idx=i)


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer & model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2WithRecurrentAttention.from_pretrained("gpt2")  # will warn about missing/unused keys
    model.config.use_cache = False  # our wrapper doesn't return cache to the block
    model.config.pad_token_id = tokenizer.eos_token_id  # silence pad warning
    model.to(device)
    model.eval()

    # Demo prompt
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.8,
        )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
