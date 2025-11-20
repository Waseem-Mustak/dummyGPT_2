# GPT-2 SMS Spam Classifier

## 🧠 How This Model Was Built
This spam classifier is based on **GPT-2**, but it wasn't just used as-is. Instead, the model  was developed from scratch and trained in **three key steps**:

1. **Custom Pretraining**: Initially trained on a book dataset to enhance text understanding.
2. **Weight Enhancement**: Integrated **OpenAI's pretrained GPT-2 weights** for better generalization.
3. **Task-Specific Fine-Tuning**: Transformed the language model into a classifier by **adding an extra layer**, **freezing select layers**, and **fine-tuning on SMS spam detection**.

Achieves **97%+ accuracy** with minimal training

## 📂 Files & Execution Flow:
```text
datasetDownload.py → Creates CSV datasets
       ↓
   dataLoader.py → Handles data processing
       ↙ ↘
fineTune.py   testPrompt.py
     (Train)      (Test)
```

## 🚀 Quick Start
### Install requirements (Python 3.8+):
<<<<<<< HEAD
Create a virtual environment and install dependencies:
=======
create a venv
>>>>>>> 92e633d0dd7b2b488c92f90b3adad4690daaaa34
```bash
pip install -r requirements.txt
```

### Run fine-tuning (5-10 mins on GPU):
```bash
python fineTune.py
```

### Automatic Progress:
1. Downloads SMS Spam Collection (746 samples)
2. Balances dataset (50% spam/ham)
3. Loads GPT-2-small (124M params)
4. Trains 6 epochs → Saves `model.pth`

### Test with custom messages:
```bash
python testPrompt.py
```

### Sample Output:
```text
"Claim your $1000 prize now!" → spam (98.7% confidence)
"See you at the meeting" → not spam (99.2% confidence)
```

## 🛠️ Key Components
### Core Files:
- `fineTune.py` - Main training script
- `testPrompt.py` - Live classification demo
- `dataLoader.py` - Batch processing & padding
- `modifiedModel.py` - GPT-2 adaptation

### Support Files:
- `datasetDownload.py` - Auto-downloads data
- `gptModel.py` - Base transformer architecture
- `loadWeight.py` - OpenAI weight integration

## ⚙️ Customization
### Change Model Size
Edit `modifiedModel.py`:
```python
# Before
CHOOSE_MODEL = "gpt2-small (124M)"

# After (for larger model)
CHOOSE_MODEL = "gpt2-medium (355M)"
```

### Training Parameters
Modify `fineTune.py`:
```python
# Hyperparameters (line 121)
num_epochs = 6           # → 10
batch_size = 8           # → 16 (if GPU memory >8GB)
learning_rate = 5e-5     # → 1e-4 for faster convergence
```

## 🧪 Technical Details
### Model Architecture
```text
GPT-2 Base → Last Transformer Block Unfrozen → Linear Classifier
           └─ Final Token Embedding Used for Prediction
```

### Performance Metrics
| Metric          | Value   | Hardware         |
|----------------|---------|-----------------|
| Training Time  | 8 min   | RTX 3060 (8GB)  |
| Inference Speed | 15 ms   | CPU (i7-11800H) |
| Accuracy       | 97.1%   | Balanced Test Set |

## ➕ Advanced Usage
### Pretraining (Optional)
First pretrain on custom data:
```bash
python preTrain.py
```
Then fine-tune as usual:
```bash
python fineTune.py
```

### Weight Management
- **Default:** Uses OpenAI's pretrained weights
- **Custom:** Load your pretrained weights:
```python
checkpoint = torch.load("custom_weights.pth")
model.load_state_dict(checkpoint["model_state_dict"])
```

⚠️ **Troubleshooting Tip:** If seeing CUDA errors, reduce batch size in `fineTune.py` (line 119).

