# Beyond Binary: Multi-Level Bengali Hate Speech Detection

## 📌 Overview

This repository contains the code and methodology for detecting multi-dimensional hate speech in Bengali social media text.

Unlike traditional classification pipelines that assign a single probability score, this project treats hate speech detection as a **Generative Extraction** problem. We utilize autoregressive Large Language Models (LLMs) fine-tuned via **QLoRA** (Quantized Low-Rank Adaptation) to simultaneously perform linguistic reasoning and output strict, machine-readable JSON dictionaries mapping three distinct dimensions:

1. **Hate Presence** (Hate Speech vs. Not Hate Speech)
    
2. **Hate Type** (e.g., Slander, Call to Violence, Religious Hate)
    
3. **Target Category** (e.g., Individual, Male, Female, Group)
    

## ✨ Key Features

- **Instruction-Based Augmentation:** Transforms raw, noisy social media text into structured prompts, preserving fragile phonetic variations and localized slang without relying on traditional, destructive data augmentation (like SMOTE or back-translation).
    
- **Structured JSON Output:** The model is trained to return deterministic JSON dictionaries, acting as an advanced linguistic parser rather than a simple classifier.
    
- **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes 4-bit NormalFloat quantization (NF4) and LoRA adapters to fine-tune massive foundation models (e.g., **Llama-3.1-8B-Instruct**) efficiently on consumer-grade GPUs (e.g., NVIDIA T4).
    
- **Unsloth Optimization:** Integrated with the Unsloth library for significantly faster training speeds and reduced VRAM consumption.
    

## 📊 Dataset (BD-SHS)

The model is trained on a highly granular Bengali hate speech dataset split into Train (80%), Validation (10%), and Test (10%) sets. The raw data requires the following columns:

- `sentence`: The raw Bengali/Banglish text.
    
- `hate speech`: Binary indicator (`1` or `0`).
    
- `type`: The specific category of hate (if applicable).
    
- `target`: The demographic target (if applicable).
    

_Note: Due to privacy and ethical constraints, the raw dataset is not included in this repository. Please ensure you place your `train.csv`, `val.csv`, and `test.csv` in the appropriate directory before running the notebook._

## ⚙️ Installation & Setup

This project is highly optimized for Google Colab or environments with NVIDIA GPUs.

**1. Clone the repository:**

Bash

```
git clone https://github.com/yourusername/bengali-hate-speech-llm.git
cd bengali-hate-speech-llm
```

**2. Install dependencies:** The project relies on specific library versions to ensure compatibility between Unsloth, TRL, and Transformers. Run the following to install the core packages:

Bash

```
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes transformers datasets
```

## 🚀 Usage

### 1. Training the Model (`project.ipynb`)

Open the provided Jupyter Notebook. The notebook handles the entire pipeline:

1. **Model Initialization:** Loads `Meta-Llama-3.1-8B-Instruct` in 4-bit precision.
    
2. **LoRA Injection:** Attaches trainable rank-decomposition matrices to the attention blocks.
    
3. **Prompt Formatting:** Maps dataset columns into the rigorous System Directive and JSON Expected Output format.
    
4. **SFT Training:** Executes Supervised Fine-Tuning using `trl.SFTTrainer`.
    
5. **Checkpointing:** Saves the optimized adapters to your specified directory.
    

### 2. Inference & Evaluation

The notebook includes an evaluation loop that switches the model to inference mode. It passes the Test dataset through the LLM, strictly parses the generated JSON strings, and records the predictions for calculating Macro F1-scores.

Python

```
# Expected Input Context
"### Input Text:\n.... ঐ ইন্দুর তোই মরছ নাই?"

# Expected Model Output
{
  "hate_presence": "Hate Speech",
  "hate_type": "Call to Violence",
  "target_category": "Individual"
}
```

### 3. Generating Loss Curves

The script automatically extracts training and validation loss from the `trainer.state.log_history` and generates a high-resolution, publication-ready loss curve (`thesis_matrices/loss_curve.png`) using `seaborn` and `matplotlib`.

## 📁 Repository Structure

Plaintext

```
├── project.ipynb           # Main training, formatting, and inference notebook
├── README.md               # Project documentation
└── thesis_matrices/        # Directory for saved high-res evaluation plots
```

## 🧠 Model Baselines

In our research, this Generative LLM approach was benchmarked against:

- **Traditional ML:** Support Vector Machines (SVM), Logistic Regression, Naive Bayes (using TF-IDF n-grams).
    
- **Standard Deep Learning:** CNN, BiLSTM, and Hybrid CNN+BiLSTM networks.
    
- **Result:** The instruction-tuned Transformers massively outperformed traditional baselines in extracting intersectional minority classes (e.g., overlapping Religious and Gender-based hate) due to deep contextual semantic embeddings.
    

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/yourusername/bengali-hate-speech-llm/issues).

## 📄 License

This project is licensed under the [MIT License](LICENSE).
