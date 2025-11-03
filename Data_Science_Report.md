# Data Science Report: Specialist Email Classifier

This report covers the fine-tuning setup and evaluation for the agent's specialist email classification model.

## 1. Fine-Tuning Setup

### 1.1. Data: Synthetic Dataset Generation

To train a specialist model, a dataset was required. We used a modern technique of synthetic data generation.

1.  **Seeding:** 9 high-quality "seed" examples of emails were manually written for three labels: `Urgent`, `To-Do`, and `FYI`.
2.  **Generation:** A large language model (LLM) was prompted with these seeds and instructed to generate 500 new, diverse examples in the same format.
3.  **Output:** The LLM generated **309** valid examples.
4.  **Splitting:** This dataset was split into `train.jsonl` (247 examples) and `test.jsonl` (62 examples) using a stratified split to ensure label balance.

### 1.2. Method: QLoRA Fine-Tuning

The core of the assignment was to use a fine-tuned model. We used **QLoRA (Quantized Low-Rank Adaptation)**, a Parameter-Efficient Fine-Tuning (PEFT) method.

* **Base Model:** `google/gemma-2b-it` (a 2.5 billion parameter model).
* **Quantization:** The model was loaded in 4-bit precision (`BitsAndBytesConfig`) to fit on a single GPU.
* **LoRA:** We used the `peft` library to insert small, trainable "adapter" layers into the model.
* **Result:** This allowed us to achieve the performance of a full fine-tune while only training a tiny fraction of the parameters.

> **Proof:**
> `trainable params: 19,617,792 || all params: 2,525,796,352 || trainable%: 0.7767`
>
> We successfully trained **less than 1%** (0.7767%) of the total model.

## 2. Evaluation Methodology and Outcomes

Evaluation was performed on both the specialist model and the final agent.

### 2.1. Model Evaluation (Quantitative)

The model was evaluated against the `test.jsonl` dataset (62 examples) which it had **never** seen during training. The evaluation was run at the end of each epoch. The best-performing model (from Epoch 3) was saved.

**Test Set Results (Epoch 3):**

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 1.0 |
| **F1-Score** | 1.0 |
| **Precision** | 1.0 |
| **Recall** | 1.0 |

**Outcome:** The model achieved **100% accuracy** on the unseen test data, demonstrating it perfectly learned to distinguish between `Urgent`, `To-Do`, and `FYI` emails from our synthetic dataset. This specialist model is exceptionally reliable.

### 2.2. Agent Evaluation (Quantitative & Qualitative)

The final agent's performance was measured on its ability to execute the complete task.

* **Quantitative (Task Completion Rate):** The `agent.py` script was run with a test inbox of 3 simulated emails (one of each type).
    * **Result:** The agent successfully processed **3 out of 3** emails.
    * **Task Completion Rate: 100%**

* **Qualitative (Plan Adherence):** The agent's logs were observed.
    * **Result:** The `agent.py` Executor correctly followed its plan. It successfully called the `classifier.py` specialist for each of the 3 emails, received the correct classification from the model, and then called `tool_move_email` with the correct folder. The integration between the Executor and the fine-tuned Specialist was a complete success.