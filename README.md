# A2: Language Model (Harry Potter Text Generator)

**Student Name:** Supanut Kompayak  
**Student ID:** st126055  
**Course:** AT82.05 Artificial Intelligence: Natural Language Understanding (NLU)

---

## 📖 Project Overview

This project implements an **LSTM-based Language Model** trained on the **Harry Potter** novel series. The model learns the writing style, vocabulary, and context of the wizarding world to generate coherent text sequences based on user prompts.

The system is deployed as a Web Application using **Flask** within a **Docker Container**, utilizing NVIDIA GPU for efficient inference.

---

## 📚 Task 1: Dataset Acquisition

### Dataset Description

For this assignment, I selected the **Harry Potter** novel series (Books 1-7) as the text corpus.

| Item | Details |
|:---|:---|
| **Dataset** | Harry Potter Complete Series (Books 1-7) |
| **Source** | [San Diego State University (SDSU)](https://dgoldberg.sdsu.edu/515/harrypotter.txt) |
| **Format** | Plain text (.txt) |
| **Characteristics** | Rich fantasy narrative with unique vocabulary (spells, names, locations) |
| **Total Samples** | ~63,040 training samples |

### Why Harry Potter?

The Harry Potter corpus is ideal for language modeling because:
- **Rich vocabulary**: Contains unique proper nouns (Hogwarts, Dumbledore, Quidditch)
- **Consistent writing style**: Single author (J.K. Rowling) ensures stylistic consistency
- **Contextual depth**: Complex narrative allows the model to learn long-range dependencies

---

## 🔬 Task 2: Model Training

### 2.1 Data Preprocessing Steps

1. **Download**: Fetched the text file from SDSU repository
2. **Lowercase conversion**: Normalized all text to lowercase
3. **Paragraph splitting**: Split text by newlines, removed empty paragraphs
4. **Train/Val/Test split**: 80% / 10% / 10% with random seed 42
5. **Tokenization**: Simple whitespace tokenizer (`text.split()`)
6. **Vocabulary building**: Created vocab with `min_freq=1` (see experiment below)
7. **Numericalization**: Converted tokens to indices using vocab mapping
8. **Batching**: Reshaped data for LSTM training (batch_size=20)

### 2.2 Model Architecture

```
LSTMModel(
  (drop): Dropout(p=0.2)
  (encoder): Embedding(9810, 200)
  (lstm): LSTM(200, 200, num_layers=2, dropout=0.2)
  (decoder): Linear(200, 9810)
)
```

| Hyperparameter | Value | Description |
|:---|:---|:---|
| **Vocabulary Size** | 9,810 | Total unique tokens |
| **Embedding Dimension** | 200 | Word vector size |
| **Hidden Dimension** | 200 | LSTM hidden state size |
| **Number of Layers** | 2 | Stacked LSTM layers |
| **Dropout** | 0.2 | Regularization rate |
| **Learning Rate** | 0.001 | Adam optimizer |
| **Scheduler** | StepLR | Decay by 0.5 every 5 epochs |
| **Epochs** | 50 | Training iterations |
| **BPTT Length** | 35 | Backpropagation through time |

### 2.3 Experiment & Analysis

I conducted an experiment to optimize the vocabulary strategy:

| Experiment | Parameter | Observation | Result |
|:---|:---|:---|:---|
| **Baseline** | `min_freq=3` | Rare words (names, spells) → `<unk>` | High `<unk>` frequency, poor coherence |
| **Optimized** ✅ | `min_freq=1` | Full vocabulary preserved | Zero/Low `<unk>`, context-aware generation |

**Conclusion:** Using `min_freq=1` allows the model to learn domain-specific vocabulary (e.g., "Hogwarts", "Dumbledore"), which is critical for generating coherent Harry Potter-style text.

### 2.4 Training Results

| Metric | Value |
|:---|:---|
| **Final Validation Perplexity** | ~5.98 |
| **Best Model Saved** | `model_v2.pth` |
| **Vocabulary Object** | `vocab_v2.pth` |

---

## 🌐 Task 3: Web Application Development

### 3.1 Application Overview

I developed a web application that allows users to interactively generate Harry Potter-style text.

| Component | Technology |
|:---|:---|
| **Backend Framework** | Flask (Python) |
| **Frontend** | HTML + CSS (Jinja2 templates) |
| **Containerization** | Docker + Docker Compose |
| **GPU Support** | NVIDIA Container Toolkit |

### 3.2 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                         │
│                    http://localhost:5000                    │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP Request (POST)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Docker Container                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    Flask App (app.py)                 │  │
│  │                                                       │  │
│  │  1. Receive user prompt from form                     │  │
│  │  2. Tokenize input text                               │  │
│  │  3. Load model (model_v2.pth) & vocab (vocab_v2.pth)  │  │
│  │  4. Run inference on GPU                              │  │
│  │  5. Generate text sequence                            │  │
│  │  6. Return result to template                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                            │                                │
│              ┌─────────────┴─────────────┐                  │
│              ▼                           ▼                  │
│     ┌──────────────┐           ┌──────────────────┐         │
│     │ model_v2.pth │           │   vocab_v2.pth   │         │
│     │ (LSTM Model) │           │ (Vocabulary Obj) │         │
│     └──────────────┘           └──────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 How the Web App Interfaces with the Model

**Request Flow:**

1. **User Input**: User types a prompt (e.g., "Harry Potter") in the input box
2. **Form Submission**: Browser sends POST request to Flask `/` endpoint
3. **Tokenization**: `simple_tokenizer()` splits the prompt into tokens
4. **Index Conversion**: Tokens are converted to indices using `vocab.stoi`
5. **Model Inference**: 
   - Input tensor is passed through the LSTM model
   - Model predicts the next token probabilities
   - Temperature sampling selects the next word
   - Process repeats for `max_words` iterations
6. **Response**: Generated text is rendered in the HTML template

**Key Code Flow (app.py):**

```python
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated = generate_text(prompt, max_words=50, temperature=0.8)
        return render_template('index.html', result=generated)
    return render_template('index.html')
```

### 3.4 UI Screenshot

![Web UI Result](Website.png)

*The model successfully generates text related to "Harry Potter" without excessive unknown tokens*

---

## 📁 Repository Structure

```
.
├── notebooks/                              # Jupyter Notebooks for training
│   └── st126055_Supanut_Kompayak_NLU_A2.ipynb
├── templates/                              # HTML templates for Flask
│   └── index.html
├── app.py                                  # Flask Application entry point
├── Dockerfile                              # Docker configuration
├── docker-compose.yml                      # Service orchestration
├── requirements.txt                        # Python dependencies
├── model_v2.pth                            # Trained LSTM Model (Best Weights)
├── vocab_v2.pth                            # Processed Vocabulary Object
├── Website.png                             # UI Screenshot
└── README.md                               # Project Documentation
```

---

## 🛠️ How to Run

### Prerequisites

- Docker & Docker Compose installed
- (Optional) NVIDIA GPU with Container Toolkit for GPU acceleration

### Steps

1. **Clone the repository:**
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd <YOUR_REPO_NAME>
   ```

2. **Build and Run with Docker Compose:**
   ```bash
   docker-compose up --build -d
   ```

3. **Access the Web Application:**
   
   Open your browser and navigate to: **http://localhost:5000**

4. **Stop the Application:**
   ```bash
   docker-compose down
   ```

---

## 📋 References

- **Dataset Source:** [SDSU Harry Potter Text](https://dgoldberg.sdsu.edu/515/harrypotter.txt)
- **PyTorch LSTM Documentation:** [pytorch.org](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- **Flask Documentation:** [flask.palletsprojects.com](https://flask.palletsprojects.com/)

---

*Submitted as part of AT82.05 Artificial Intelligence: Natural Language Understanding (NLU), Asian Institute of Technology (AIT).*
