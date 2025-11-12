

# Mercor AI Text Detection - 99.23% Accuracy Solution

This repository contains the code for a high-performing solution to the Mercor AI Text Detection Kaggle competition. The final model achieves a **99.23% ROC-AUC** on the validation set by ensembling two powerful transformer models.

## Project Overview

The goal of this competition was to distinguish between human-written and AI-generated text. This solution tackles the problem by leveraging the principle that a diverse committee of expert models often outperforms any single model. By combining a specialized AI text detector with a state-of-the-art general NLP model, we can capture a wider range of linguistic patterns indicative of AI-generated content.

![Final Prediction Distribution](https://i.imgur.com/example.png) <!-- Replace with a link to your plot image -->
![Model Correlation](https://i.imgur.com/example2.png) <!-- Replace with a link to your plot image -->

## Key Features

-   **Ensemble Modeling:** Combines predictions from multiple models to improve accuracy and robustness.
-   **State-of-the-Art Transformers:** Utilizes `fakespot-ai/roberta-base-ai-text-detection-v1` and `microsoft/deberta-v3-base`.
-   **Data Preprocessing:** Includes text cleaning and minority-class upsampling to create a balanced and clean dataset.
-   **Efficient Training:** Employs gradient accumulation, BF16 mixed-precision, and early stopping to manage memory and prevent overfitting.
-   **Advanced Blending:** Uses a hybrid of weighted averaging (based on validation scores) and rank averaging for the final submission.

## The Winning Pipeline

The solution follows a structured, multi-stage pipeline:

1.  **Data Preparation:**
    -   Load the `train.csv` and `test.csv` datasets.
    -   Clean text fields by removing extra whitespace and newlines.
    -   Create a unified input format: `TOPIC: [topic_text]\n\nANSWER: [answer_text]`.
    -   Balance the training data by upsampling the minority class to prevent model bias.

2.  **Model Training:**
    -   A generic training function fine-tunes two separate models on the prepared data:
        -   **Model 1 (Specialist):** `fakespot-ai/roberta-base-ai-text-detection-v1`
        -   **Model 2 (Generalist):** `microsoft/deberta-v3-base`
    -   Training is optimized using memory-saving techniques like smaller batch sizes and gradient accumulation, particularly for the larger DeBERTa model.

3.  **Prediction & Ensembling:**
    -   Each trained model predicts the probability of "cheating" on the test set.
    -   The predictions are combined using a two-stage blending process:
        1.  **Weighted Average:** Predictions are averaged, with weights proportional to each model's validation ROC-AUC score.
        2.  **Rank Average:** Predictions are converted to ranks, averaged, and scaled back to a 0-1 range. This makes the ensemble robust to outliers.
    -   The final submission is a blend of these two strategies (`60% weighted_avg + 40% rank_avg`).

4.  **Submission:**
    -   The final blended probabilities are saved to `submission.csv` in the required format.

## How to Run the Code

This project is designed to run in a Kaggle Notebook environment with a GPU.

**1. Dependencies:**
The code relies on standard data science libraries and the Hugging Face ecosystem.
```bash
pip install pandas numpy scikit-learn torch transformers datasets sentencepiece
```

**2. Dataset:**
-   Ensure the competition dataset is located at `/kaggle/input/mercor-ai-detection/`.
-   The directory should contain `train.csv` and `test.csv`.

**3. Execution:**
-   Place the entire Python script into a single cell in a Kaggle Notebook.
-   Ensure the GPU accelerator is turned on.
-   Run the cell. The script will automatically handle data loading, preprocessing, training both models, ensembling, and generating the final `submission.csv` file in the `/kaggle/working/` directory.

## What Could Be Done Better?

While this approach was highly successful, further improvements could be explored:
-   **Incorporate More Models:** Adding a third or fourth diverse model (like ELECTRA or a T5-based model) could further enhance the ensemble's power.
-   **Cross-Validation:** Instead of a single train/validation split, using k-fold cross-validation would produce more robust validation scores and out-of-fold predictions for ensembling.
-   **Hyperparameter Tuning:** A more exhaustive search for hyperparameters (learning rate, batch size, etc.) for each model could unlock additional performance.
