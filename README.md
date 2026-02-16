<p align="center">
  <img src="./assets/logo.webp" width="200" alt="CmiLab Logo">
</p>

# Scalable Machines With Intrinsic Higher Mental States

<p align="center">
  <a href="https://cmilab.org/aichip/trend/">
    <img src="https://img.shields.io/badge/🌐-Project_Website-blue?style=for-the-badge" alt="Website">
  </a>
</p>

**Status:** First reference implementation of Cooperative Context-sensitive Cognitive Computation (Co⁴).
The architecture is under active development as part of the TREND project.


## 🌐 Overview
This repository provides a general reference implementation of Scalable Machines With Intrinsic Higher Mental States, extending prior work (adeel, 2025) with an open, scalable Cooperative Context-sensitive Cognitive Computation (Co<sup>4</sup>) architecture and early validation on large-scale benchmarks such as ImageNet-1K. 

This codebase is intentionally presented as a research platform rather than a finalized or optimized model. It is designed to support experimentation, understanding, reproducibility of basic results, and community-driven exploration of work that goes beyond conventional AI toward real understanding.

## 🧠 Core idea

Modern Transformers determine relevance after attention, relying on depth and quadratic computation. Co⁴ introduces intrinsic higher mental states that allow the model to:

- generate internal predictions to pre-select relevant information before attention via triadic neuronal-level modulation loops
- enforce pre-reflective contextual coherence at the representation level
- enable faster learning with reduced computational demand (e.g., fewer heads, layers, and tokens)
- reduce computation from O(N²) to near-linear scaling in N

The mechanism is inspired by pyramidal two-point neurons (TPNs) and implemented through triadic modulation loops among Q, K, and V latent populations.

This repository provides the first open reference implementation of this architecture.

## 📊 Reproducing key results

The repository includes scripts for reproducing experiments reported in the paper:

- CIFAR-10
- Tiny-ImageNet
- Mini-ImageNet
- ImageNet-1K (early scaling)
- CartPole
- PyBullet Ant
- Acrobot
- MountainCar
- CarRacing

These correspond to:

- Figure 4 (vision experiments)
- Figure 5 (RL experiments)
- Table 1–4 (ablation and scaling results)


## 🏗️ Architecture
![System Architecture](./assets/Co4.png)

## Latent Triadic Modulation Mechanism

Latent tokens $Q_L$, $K_L$, and $V_L$ are initialized from a random distribution and used as feedforward (FF) inputs or receptive fields (R). Contextual input (context-modulated prediction), $Q^{N-1}_m$, $K^{N-1}_m$, $V^{N-1}_m$, and μ act as proximal (P), distal (D), and universal (U) context (C), providing feedback (FB). For the first layer, these are equal to: $Q_X$, $K_X$, $V_X$, and initialized μ. The TPN-like circuits $Q_m$, $K_m$, and $V_m$ evolve via asynchronous triadic Modulation Transfer Functions (AMTFs) under Apical Drive (AD) and Apical Drive + Awake Thought (AD + Awake) states. The evolved latent tokens $Q_m$, $K_m$, and $V_m$ are then selected and fed into the self-attention block.

## Gradient Flow  
A demonstration of how different modulatory cooperation laws Φ(𝑅,𝐶) reshape the cooperation surface and its gradient field ∇Φ(𝑅,𝐶) over the 𝑅−𝐶. Changes in contextual and receptive-field strength move the system between apical isolation, apical amplification, apical drive, and AD+Awake regimes, producing corresponding deformations in the geometry of gradient flow. By shaping representations prior to attention, these modulation laws guide gradients along coherent RF–CF interaction manifolds, reducing propagation through noisy or irrelevant directions. This structured learning geometry helps explain the faster convergence and improved learning efficiency observed in Co<sup>4</sup> compared to standard Transformers, where gradients propagate without such context-conditioned modulation.

**Click the GIF below to open the interactive demo in your browser**
[![GradientFlowDynamics](https://github.com/user-attachments/assets/808933c9-ef82-4e75-a561-c49150f55ed6)](https://aria-funded-trend.github.io/HMS.github.io/)

<!--[![GradientFlowDynamics](https://github.com/user-attachments/assets/808933c9-ef82-4e75-a561-c49150f55ed6)](https://aria-funded-trend.github.io/HMS.github.io/)
-->


## Object Classification 
![Attention_Maps](./assets/attention_heatmaps3.png)
Single-layer Co4 vs. standard attention-only Vision Transformer (ViT) (Dosovitskiy, 2020) trained on Mini-ImageNet for 30 epochs. Visualization of the complete attention distribution over _N_ input tokens, showing more scattered attention and less selective localization in the ViT. In contrast, Co4 shows more centered, context-aware activation patterns. Notably, no top-k pooling was applied.

![Comparison](./assets/bird.png)
_Top-k_ pooling: Results of _top-k_ token selection in the standard Transformer are shown in the ablation study (Table 4 of the paper), where reducing the number of generated tokens via the _top-k_ feature leads to significant improved performance (faster learning) in Co4, whereas applying the same _top-k_ feature approach in attention-only Vision Transformer (ViT) reduces performance. This is early training comparison between an ViT _(Dosovitskiy, 2020)_ trained from scratch, and a Co<sup>4</sup> machine endowed with intrinsic higher mental states that pre-select relevant information before attention is applied. The task is to identify a bird from the Mini-ImageNet dataset. In the ViT model, the brightness indicates regions highlighted after applying attention. In contrast, Co<sup>4</sup> highlights important regions _(top k latent tokens)_ using internally generated awake imaginative  states _before_ attention is applied. Co<sup>4</sup> exhibits earlier and sharper activation over the semantically relevant object (bird), indicating more coherent internal inference. Co⁴ learns faster early, but scaling behavior is still being explored. These results raise questions about the necessity of attention and latent pooling, suggesting a natural convergence toward O(N) complexity.}

## Reinforcement Learning
### 🎥 Demo

<!--<video src="https://github-production-user-asset-6210df.s3.amazonaws.com/122742805/546751693-1b619473-1e64-405b-ae85-c0d4dc1ea571.mp4"
       controls
       muted
       loop
       playsinline
       width="100%">
</video>
-->


https://github.com/user-attachments/assets/892a8595-3071-4b2a-a9c7-d8e151ec111d






Comparison of a permutation-invariant Transformer (left) and the CO4 model with intrinsic higher-order mental states (right), both trained for 100 episodes; CO4 reaches ~700 reward while transformer only reaches 245 reward.


## 📄 License
The source code is released under the [Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license, permitting reuse and modification for research and academic purposes while restricting commercial use — see the [LICENSE](LICENSE) file for details.

## BibTeX
@article{adeel2025beyond,
  title   = {Beyond Attention: Toward Machines with Intrinsic Higher Mental States},
  author  = {Adeel, Ahsan},
  journal = {arXiv preprint arXiv:2505.06257},
  year    = {2025}
}

@article{adeel2026scalable,
  title   = {Scalable Machines With Intrinsic Higher Mental States},
  author  = {Adeel, Ahsan},
  journal = {arXiv preprint},
  note    = {arXiv submission (in preparation)},
  year    = {2026}
}

