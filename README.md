<p align="center">
  <img src="./assests/logo.webp" width="200" alt="CmiLab Logo">
</p>

# Scalable Machines With Intrinsic Higher Mental States

<p align="center">
  <a href="https://cmilab.org/aichip/">
    <img src="https://img.shields.io/badge/🌐-Project_Website-blue?style=for-the-badge" alt="Website">
  </a>
</p>

## 🌐 Overview
This repository provides a general reference implementation of **Scalable Machines With Intrinsic Higher Mental States**, extending prior work with an open, scalable architecture (CO4) and early validation on large-scale benchmarks such as ImageNet-1K. The model achieves near-linear scaling with respect to input size while enabling faster learning and reduced computational cost by using fewer heads, layers, and tokens.

This codebase is intentionally presented as a research platform rather than a finalized or optimized model. It is designed to support experimentation, understanding, and community-driven exploration of alternative architectural directions beyond standard attention mechanisms.

## 🏗️ Architecture
![System Architecture](./assests/Architecture.png)

## Reinforcement Learning
### 🎥 Demo
<p align="center">
  <img width="100%" src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXdkZzNkdGY3ZjQ5eDdtcDJic3J4N3RhaHR5MGpoZ2JoNjcwcG4zeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/n0m3XuzrmgKpowAdyi/giphy.gif" alt="Project Demo">
</p>

Comparison of a permutation-invariant Transformer (left) and the CO4 model with intrinsic higher-order mental states (right), both trained for 100 episodes; CO4 reaches ~700 reward.

## Object Classification 
![Comparison]()

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.