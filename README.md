# English POS Tagger

This project is an implementation of an English Part-of-Speech (POS) tagger using the Viterbi algorithm. The POS tagger is designed to label each word in a sentence with its corresponding part of speech (e.g., noun, verb, adjective).

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)


## Introduction

The POS tagger uses the Viterbi algorithm, which is a dynamic programming algorithm used for finding the most likely sequence of hidden states (in this case, POS tags) that result in a sequence of observed events (words). The model is trained on a tagged corpus and then used to predict the POS tags for new sentences.

## Features

- **Efficient POS tagging:** Utilizes the Viterbi algorithm for efficient and accurate POS tagging.
- **Customizable:** Can be retrained with different corpora to improve accuracy for specific domains.
- **Easy to use:** Simple interface for tagging new sentences.

## Installation

To install and run the POS tagger, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/english-pos-tagger.git
   cd english-pos-tagger
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the POS tagger, you can check my app 

```bash
python pos_tagger.py "Your sentence here."
```

Example:

```bash
python pos_tagger.py "The quick brown fox jumps over the lazy dog."
```

This will output the sentence with each word tagged with its corresponding part of speech.

### Training the Model

If you want to train the model with a different corpus, just pass the dataset into HMM model in the `code.model`
```python
from code.model import HMM
import pickle
model = HMM(dataset)
with open('/kaggle/working/hmm_model.pkl','wb') as f:
    pickle.dump(model,f)
```
The dataset structure being:
```
data.json
[
    {
        "index"->int:{0,1,2...},
        "sentence"->list[str]:{["word",...]}
        "labels"->list[str]:{["VB","NN"]}
    },
    ...
]
```
## File Structure

```
ENGLISH_POS_TAGGER/
├── code/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── viterbi_decoding.py
├── input/
│   ├── dev.json
│   └── train.json
├── app.py
├── english-pos-tagger-viterbi.ipynb
├── hmm_model.pkl
├── README.md
└── requirements.txt

```

## Dependencies

- numpy
- pandas
- pickle

To install the dependencies, run:

```bash
pip install -r requirements.txt
```



Thank you for using the English POS Tagger! If you have any questions or issues, please open an issue in the repository.