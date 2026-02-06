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
This repository provides a general reference implementation of **Scalable Machines With Intrinsic Higher Mental States**, extending prior work with an open, scalable architecture (CO4) and early validation on large-scale benchmarks such as ImageNet-1K. The model achieves near-linear scaling with respect to input size while enabling faster learning and reduced computational cost by using fewer heads, layers, and tokens.

This codebase is intentionally presented as a research platform rather than a finalized or optimized model. It is designed to support experimentation, understanding, and community-driven exploration of alternative architectural directions beyond standard attention mechanisms.

## 🏗️ Architecture
![System Architecture](./assests/Co41.png)

Latent QL, KL, and VL tokens are initialized from a random distribution and used as feedforward (FF) input or receptive field (RF). Input QX, KX, VX, and µ act as proximal (P), distal (D), and universal (U) contextual factors (CFs) providing feedback (FB). The Qm, Km, and Vm TPN-like circuits evolve via asynchronous triadic Modulation Transfer Functions (AMTF) under apical drive and apical drive + awake thought states. The evolved Qm, Km, and Vm are then selected and fed into the self-attention block.


## Gradient Flow  
The plot shows the interactive gradient flow of the cooperation dynamics defined by Cooperation(R, C) = R² + 2R + C(1 + |R|) and Cooperation(R, C) = C² + 2C + C(1 + |R|) revealing distinct bursting regimes that emerge from evidence–context coupling and their gradient flow.

[![GradientFlowDynamics](https://github.com/user-attachments/assets/808933c9-ef82-4e75-a561-c49150f55ed6)](https://beingtalha.github.io/ARIA-Funded-TREND.github.io/)



## Object Classification 
![Comparison](./assests/bird.png)
Early training comparison between an attention-only Vision Transformer (ViT) (Dosovitskiy, 2020), trained from scratch, and a CO4 machine endowed with intrinsic higher mental states that pre-select relevant information before attention is applied, in a task to identify a bird from the Mini-ImageNet dataset. The brightness in the attention-only Transformer highlights areas after applying attention, whereas the CO4 model first highlights important regions in the image using its internal mechanisms, before attention is applied.

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

