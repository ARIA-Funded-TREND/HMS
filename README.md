<p align="center">
  <img src="./assests/logo.webp" width="200" alt="CmiLab Logo">
</p>

# Scalable Machines With Intrinsic Higher Mental States

<p align="center">
  <a href="https://cmilab.org/aichip/trend/">
    <img src="https://img.shields.io/badge/🌐-Project_Website-blue?style=for-the-badge" alt="Website">
  </a>
</p>

## 🌐 Overview
This repository provides a general reference implementation of **Scalable Machines With Intrinsic Higher Mental States**, extending prior work _(adeel, 2025)_ with an open, scalable Cooperative Context-sensitive Cognitive Computation (Co<sup>4</sup>) architecture and early validation on large-scale benchmarks such as ImageNet-1K. The model achieves near-linear scaling with respect to input size while enabling faster learning and reduced computational cost by using fewer heads, layers, and tokens.

This codebase is intentionally presented as a research platform rather than a finalized or optimized model. It is designed to support experimentation, understanding, and community-driven exploration of alternative architectural directions beyond standard attention mechanisms.

## 🏗️ Architecture
![System Architecture](./assests/Architecture.png)

## Latent Triadic Modulation Mechanism

Latent tokens $Q_L$, $K_L$, and $V_L$ are initialized from a random distribution and used as feedforward (FF) inputs or receptive fields (R). Tokens $Q_X$, $K_X$, $V_X$, and $\mu$ act as proximal (P), distal (D), and universal (U) contextual fields (CFs), providing feedback (FB) on the fly. The TPN-like circuits $Q_m$, $K_m$, and $V_m$ evolve via asynchronous triadic Modulation Transfer Functions (AMTFs) under Apical Drive (AD) and Apical Drive + Awake Thought (AD + Awake) states. The evolved latent tokens $Q_m$, $K_m$, and $V_m$ are then selected and fed into the self-attention block.

## Gradient Flow  
A demonstration of how different modulatory cooperation laws Φ(𝑅,𝐶) reshape the cooperation surface and its gradient field ∇Φ(𝑅,𝐶) over the 𝑅−𝐶. Changes in contextual and receptive-field strength move the system between apical isolation, apical amplification, apical drive, and AD+Awake regimes, producing corresponding deformations in the geometry of gradient flow. By shaping representations prior to attention, these modulation laws guide gradients along coherent RF–CF interaction manifolds, reducing propagation through noisy or irrelevant directions. This structured learning geometry helps explain the faster convergence and improved learning efficiency observed in Co<sup>4</sup> compared to standard Transformers, where gradients propagate without such context-conditioned modulation.

**Click the GIF below to open the interactive demo in your browser**
[![GradientFlowDynamics](https://github.com/user-attachments/assets/808933c9-ef82-4e75-a561-c49150f55ed6)](https://aria-funded-trend.github.io/HMS/)

<!--[![GradientFlowDynamics](https://github.com/user-attachments/assets/808933c9-ef82-4e75-a561-c49150f55ed6)](https://aria-funded-trend.github.io/HMS.github.io/)
-->


## Object Classification 
![Comparison](./assests/bird.png)
Early training comparison between an attention-only Vision Transformer (ViT) _(Dosovitskiy, 2020)_ trained from scratch, and a Co<sup>4</sup> machine endowed with intrinsic higher mental states that pre-select relevant information before attention is applied. The task is to identify a bird from the Mini-ImageNet dataset. In the ViT model, the brightness indicates regions highlighted after applying attention. In contrast, Co<sup>4</sup> highlights important regions _(top k latent tokens)_ using internally generated awake imaginative  states _before_ attention is applied. Co<sup>4</sup> exhibits earlier and sharper activation over the semantically relevant object (bird), indicating more coherent internal inference.

## Reinforcement Learning
### 🎥 Demo

<video src="https://github-production-user-asset-6210df.s3.amazonaws.com/122742805/545188999-4373bd43-49a4-495e-881a-697671bd48ca.mp4"
       controls
       muted
       loop
       playsinline
       width="100%">
</video>


Comparison of a permutation-invariant Transformer (left) and the CO4 model with intrinsic higher-order mental states (right), both trained for 100 episodes; CO4 reaches ~700 reward while transformer only reaches 245 reward.


## 📄 License
The source code is released under the [Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license, permitting reuse and modification for research and academic purposes while restricting commercial use — see the [LICENSE](LICENSE) file for details.

## BibTeX
@article{adeel2025beyond,
  title={Beyond Attention: Toward Machines with Intrinsic Higher Mental States},
  author={Adeel, Ahsan},
  journal={arXiv preprint arXiv:2505.06257},
  year={2025}
}
