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
![System Architecture](./assests/Architecture.png)

latent QL, KL, and VL tokens are initialized from a random distribution and used as feedforward (FF) input or receptive field (RF). Input QX, KX, VX, and µ act as proximal (P), distal (D), and universal (U) contextual factors (CFs) providing feedback (FB). The Qm, Km, and Vm TPN-like circuits evolve via asynchronous triadic Modulation Transfer Functions (AMTF) under apical drive and apical drive + awake thought states. The evolved Qm, Km, and Vm are then selected and fed into the self-attention block.


## Reinforcement Learning
### 🎥 Demo
<p align="center">
  <img width="100%" src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbTViZ3V2NzlueDB4bDkwNHg3YjFzazRqYXA2Ym84NXgxNGdiY2o5YSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pcruZRhUo6qyPVD0N9/giphy.gif" alt="Project Demo">
</p>


Comparison of a permutation-invariant Transformer (left) and the CO4 model with intrinsic higher-order mental states (right), both trained for 100 episodes; CO4 reaches ~700 reward while transformer only reaches 245 reward.

## Object Classification 
![Comparison](./assests/bird.png)
Early training comparison between an attention-only Vision Transformer (ViT) (Dosovitskiy, 2020), trained from scratch, and a CO4 machine endowed with intrinsic higher mental states that pre-select relevant information before attention is applied, in a task to identify a bird from the Mini-ImageNet dataset. The brightness in the attention-only Transformer highlights areas after applying attention, whereas the CO4 model first highlights important regions in the image using its internal mechanisms, before attention is applied.
## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.