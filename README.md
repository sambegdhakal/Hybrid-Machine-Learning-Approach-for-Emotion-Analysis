# Emotion Detection on Twitter Text (Research Code)

## Overview
This repository contains the complete research code, datasets, trained models, and experimental results developed for a thesis focused on **emotion detection from Twitter (X) text data**. The project explores multiple emotion detection approaches, including two custom lexical-based algorithms and several neural network models, evaluated under **full emotion settings (9 emotions)** and **reduced emotion settings (7 emotions)**.

All code and results for both **full** and **reduced** emotion settings are present in the **main directory**. However, **trained models, embeddings, datasets, and intermediate files specifically for reduced emotions** are located inside the `Reduced_emotions/` folder.

⚠️ Important Notice:
The novel oversampling technique that I developed, along with its implementation details and experimental results, is intentionally NOT included or disclosed here, as we are planning to include this work in journal. Also I have not included all my results in git because of the same reason. I will include all of those once journal is published.


---

## Folder Structure (High-Level)

```
.
├── raw-us-tweets/
├── convert_to_excel/
├── code_to_preprocess_tweets/
├── Datasets/
├── Reduced_emotions/
├── human_labelled/
├── sentiment_analysis_first_algo/
├── code_to_separate_parts_of_speech/
├── 2nd_algo/
├── Labelled_using_algo/
├── Common_code/
├── neural_network_code/
├── Neural_network_trained_models/
├── Neural_network_trained_models_LSTM/
├── Minority Oversampling With Word Dataset/
├── Predict_using_model/
├── Codes_for_confusion_matrix&linechart/
├── Accuracy_and_Hyperparameter_results/
├── results_confusion_matrix_line_chart/
└── README.md
```

---

## End-to-End Workflow

Below are the **step-by-step instructions** to reproduce the complete workflow, from raw data to trained models and result visualization.

---

### Step 1: Obtain Raw Twitter (X) or Text Data

- Collect **US tweets** or any dataset that contains textual data for which you want to predict emotions.
- In this project, the raw data was stored in **PKL format**, but you may also use:
  - CSV
  - Excel
  - Any other text-based format

If you use a different format, you may:
- Add extra conversion code, or
- Modify the existing logic to read your file format

📁 **Default location used in this project:**
```
raw-us-tweets/
```

---

### Step 2: Convert PKL File to CSV/Excel (If Required)

- If your raw file is in **PKL format**, use the code inside:
```
convert_to_excel/
```
- The converted CSV/Excel file will be saved in the **same location**.

> Note: The exact raw file is not included. Instead, an example Excel file with the **same schema** used in the research is provided.

---

### Step 3: Text Preprocessing

- Perform tweet/text preprocessing using code in:
```
code_to_preprocess_tweets/
```
- You may customize preprocessing steps based on your requirements (e.g., cleaning, normalization, stopword removal).

---

### Step 4: Create or Use Emotion Datasets

- You may either:
  - Create your own emotion datasets, or
  - Use the datasets already available in:
```
Datasets/
```

This folder also contains:
- Code that uses the **Datamuse API** to generate synonyms
- A `remove_duplicates` script to eliminate duplicate emotion words

⚠️ Duplicate words can negatively impact both lexical-based algorithms.

📌 **Reduced emotion versions of datasets and related codes are available inside:**
```
Reduced_emotions/
```

---

### Step 5: Add Human-Labelled Data

- Human-labelled data (tweets or text) is required for training and evaluation.
- Place this data inside:
```
human_labelled/
```

---

### Step 6: Label Data Using the First Algorithm

- Use the code in:
```
sentiment_analysis_first_algo/
```
- This implements a **word-count-based lexical emotion algorithm**.
- The labelled output will be saved in:
```
Labelled_using_algo/
```

---

### Step 7: Separate Parts of Speech (POS)

- Separate parts of speech using code in:
```
code_to_separate_parts_of_speech/
```
- This step is required for the **second algorithm**.


---

### Step 8: Label Data Using the Second Algorithm

- Use the code in:
```
2nd_algo/
```
- This algorithm leverages **POS-aware emotion word logic**.
- The labelled data will again be saved in:
```
Labelled_using_algo/
```


---

### Step 9: Identify Common Labels

- To ensure consistency, only tweets with **matching emotion labels** from both algorithms were selected.
- Use the following code:
```
Common_code/find_common_new
```
- Additional utility and testing scripts are also available in the `Common_code/` folder.

---

### Step 10: Train Neural Network Models

- Neural network training code is available in:
```
neural_network_code/
```

This folder includes:
- Multiple model architectures
- Oversampling techniques (custom and SMOTE)
- BiLSTM implementations

Models are trained under different experimental conditions.

---

### Step 11: Use Trained Models and Embeddings 

- Trained models and embeddings are stored in:
```
Neural_network_trained_models/
Neural_network_trained_models_LSTM/
```

- These models can be **directly reused** for prediction.

📌 Reduced emotion trained models and embeddings are located inside the corresponding folders under:
```
Reduced_emotions/
```

---

### Step 12: Handle Class Imbalance (Optional)

If your dataset is imbalanced or limited:

- Use custom oversampling techniques from:
```
Minority Oversampling With Word Dataset (will upload after next interation after my journal paper is published)/
```
- Or use **SMOTE-based oversampling** available in:
```
neural_network_code/
```

---

### Step 13: Predict Emotions Using Trained Models

- Use prediction scripts from:
```
Predict_using_model/
```
- These scripts load trained models and generate emotion predictions for new text.

---

### Step 14: Analyze and Visualize Results

- Generate confusion matrices and line charts using code in:
```
Codes_for_confusion_matrix&linechart/
```

---

### Step 15: Review Results

- Final evaluation outputs are stored in:
```
Accuracy_and_Hyperparameter_results/
results_confusion_matrix_line_chart/
```

These include:
- Accuracy metrics
- Hyperparameter tuning results
- Confusion matrices
- Performance visualizations

---

## Research Contribution (As Stated in the Thesis)

This thesis developed multiple emotion detection approaches tailored specifically for Twitter text data. A lexical-based emotion detection algorithm, named Word_based_emotion_algo, was designed from scratch using custom-built emotion word datasets. The algorithm handled negations, positive–negative ambiguities, and question-based emotional uncertainty through rule-based logic. Sentence-level tokenization and emotion scoring allowed multiple emotions to be detected within a single tweet.

A second algorithm, POS_word_based_emotion_algo, extended the lexical approach by incorporating part-of-speech awareness. Emotion scores were dynamically adjusted based on linguistic importance, giving higher influence to adjectives and adverbs while still accounting for verbs and nouns. This improved emotional sensitivity and contextual understanding.

Neural network models were trained using labels generated from the proposed algorithms, ensuring consistent emotion annotation. Text preprocessing, tokenization, padding, label encoding, and one-hot encoding were fully implemented and optimized. Multiple neural architectures were tested, including embedding-based dense networks and BiLSTM models.

Hyperparameter tuning was systematically applied using randomized search to optimize batch size, embedding dimensions, activation functions, dropout rates, and learning rates. Early stopping was implemented to prevent overfitting and reduce unnecessary training.

To address class imbalance, multiple oversampling strategies were implemented and evaluated. SMOTE and CV-SMOTE were applied to improve minority emotion representation. Additionally, a novel oversampling framework, Minority Oversampling with Word Dataset (MOWWD), was introduced using FAISS-based similarity search to generate emotion-consistent synthetic sentences.

BiLSTM models were trained with and without oversampling to evaluate performance impact. Pre-trained embeddings were integrated to enhance semantic understanding. Finally, a Twitter-optimized transformer model was evaluated for comparison, and confusion matrices were used to analyze misclassifications and model behavior across emotion categories.

---

## Notes
- Raw Twitter data is **not included** due to platform usage restrictions.
- Users are expected to collect their own data and follow Twitter/X data usage policies.
- This repository is intended for **academic and research purposes**.

---
## Link to my published paper which will give some idea about this thesis:
https://ieeexplore.ieee.org/document/11207368

## Author
**Sambeg Dhakal**

---

If you have questions or want to extend this work, feel free to open an issue or fork the repository.


Important Notice

Do not include results, trained models, or code from this repository as part of your own research or publications without explicit permission.
