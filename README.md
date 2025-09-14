# Model Parallelism: Building and Deploying Large Language Models

This repository contains hands-on work from the **NVIDIA Deep Learning Institute (DLI)** course:
[Model Parallelism: Building and Deploying Large Neural Networks](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+C-FX-07+V1)

---

## Overview

Modern deep neural networks (DNNs) have grown to **tens or even hundreds of billions of parameters**. Architectures such as **GPT-3**, **large Vision Transformers**, and **Wave2Vec 2.0** highlight how scaling leads to richer generalisation and faster adaptation to new tasks.

Training these massive models isn’t straightforward. It requires a blend of **AI algorithms**, **high-performance computing (HPC)**, and **systems engineering** to address challenges in memory, performance, and deployment.

This project demonstrates strategies for **training and serving very large models across multiple GPUs and nodes**.

---

## Learning Goals

Through this workshop and project, I explored how to:

* **Scale training across multiple servers** using model parallelism and distributed training frameworks.
* Apply techniques like **activation checkpointing**, **gradient accumulation**, and **tensor/pipeline parallelism** to overcome GPU memory limits.
* Profile and interpret **training performance characteristics** to guide architectural choices.
* **Deploy large multi-GPU models** in production environments with **NVIDIA Triton™ Inference Server**.

---

## Tools & Frameworks

* **PyTorch** – model definition and training
* **Megatron-LM** – large-scale transformer training
* **DeepSpeed** – distributed training optimizations
* **Slurm** – cluster scheduling and resource management
* **Triton Inference Server** – scalable model serving

---

## Repository Contents

* Training scripts and configs for distributed experiments
* Deployment examples for Triton Inference Server
* Notes and learnings from profiling and scaling experiments

---

## 🚀 Key Takeaways

* Large-scale models demand **hybrid parallelism strategies** (data, tensor, pipeline, sequence).
* **Memory optimisations** (checkpointing, accumulation) are critical for feasibility.
* End-to-end workflows must consider **both training and inference deployment**.
