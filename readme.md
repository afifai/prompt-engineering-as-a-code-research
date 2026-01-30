# 🛡️ JagaPesan: LLMOps for Spam Detection

> **Prompt Engineering as Code:** An automated CI/CD pipeline for AWS Bedrock Agents that treats AI Prompts like software code.

![CI Status](https://img.shields.io/badge/LLMOps-Active-green)
![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange)
![Model](https://img.shields.io/badge/Model-Claude%203%20Sonnet-purple)

## 📖 Overview

**JagaPesan** is an intelligent SMS Assistant Agent built on **AWS Bedrock**. It classifies incoming Indonesian messages into **SPAM**, **SAFE**, or **OPERATORS**, provides reasoning, and generates context-aware replies (sarcastic for spam, polite for humans).

This repository demonstrates a mature **LLMOps workflow**:
1.  **Prompts are Code:** The agent's instruction is version-controlled.
2.  **Automated Evaluation:** Every Pull Request triggers a comparison test.
3.  **Regression Testing:** Ensures new prompts don't break existing logic.
4.  **Live Reporting:** The bot posts a detailed "Impact Report" directly to the PR.

---

## 🤖 Features

* **Smart Classification:** Distinguishes between Fraud (Spam), Personal Chat (Safe), and Promo (Operators).
* **Auto-Reply Generator:**
    * 👻 **Spam:** Generates witty/sarcastic replies to troll scammers.
    * 👤 **Safe:** Generates polite & casual replies.
    * 🏢 **Operator:** No reply logic.
* **Localized Reasoning:** Provides analysis in Indonesian context.
* **Dual-Dataset Evaluation:**
    * `validation.csv`: Labeled data for calculating Accuracy & Regressions.
    * `inference.csv`: Unlabeled data for testing generative behavior (Live Demo).

---

## 🛠️ Architecture & Workflow

The pipeline is defined in `.github/workflows/pipeline.yml`.

```mermaid
graph LR
    A[Dev modifies prompt] -->|Push| B(Pull Request)
    B --> C{GitHub Action}
    C -->|Step 1| D[Eval Main Branch]
    C -->|Step 2| E[Eval PR Branch]
    D & E --> F[Compare Metrics]
    F --> G[Post Comment Report]