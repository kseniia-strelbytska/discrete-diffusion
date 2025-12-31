## Project Overview

An ongoing investigation into **rule extrapolation** and **length generalisation** in **Discrete Diffusion Models** on out-of-distribution formal language prompts.

This work is inspired by recent advances in structured generative modelling, particularly:

- **Sequence Modelling with Discrete Diffusion**  
  https://arxiv.org/pdf/2406.07524

- **Compositional Generalisation on OOD Prompts in Language Models** (NeurIPS 2024)  
  https://proceedings.neurips.cc/paper_files/paper/2024/file/3d9ef68629089da055334c2d41dfcf93-Paper-Conference.pdf

---

## Core Objectives

1. Train **transformer-based discrete diffusion models** on formal languages defined as the **intersection of two rules**, and analyse whether and how these rules are learned.
2. Evaluate **out-of-distribution generalisation** on prompts that violate one rule, and compare performance against **autoregressive transformer baselines**.
3. Investigate **length generalisation** by testing model performance on sequences longer than those seen during training.

---

## Motivation

Generalisability is a key challenge in modern generative modelling. Understanding whether models can **extrapolate abstract rules**, rather than merely interpolate training distributions, is critical for robust reasoning and compositional generalisation in neural sequence models.
