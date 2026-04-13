# 🚀 Project Templates

Ready-to-use project structures for building AI applications with Hugging Face models.

## Available Templates

### For Developers

| Template | Description | Use Case |
|----------|-------------|----------|
| API Service | FastAPI service template | Deploy models as REST API |
| Chatbot | Conversational AI template | Build chatbots |
| Data Pipeline | ML data processing | Process and analyze data |
| Fine-tuning Project | Custom training setup | Fine-tune models |

### For Vibe Coders

| Template | Description | Use Case |
|----------|-------------|----------|
| AI Playground | Interactive chat UI | Experiment with models |
| Image Generator | Text-to-image app | Create AI art |
| Summarizer | Document summarizer | Quick summaries |

## API Service

Purpose: Deploy Hugging Face models as a REST API

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
sentiment = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze(input: TextInput):
    result = sentiment(input.text)
    return {"text": input.text, "sentiment": result}
```

## Chatbot

Purpose: Build a conversational AI chatbot

```python
class Chatbot:
    def __init__(self, model_id="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.history = []
    
    def chat(self, message):
        self.history.append({"role": "user", "content": message})
        inputs = self.tokenizer("\n".join(
            [f"{m['role']}: {m['content']}" for m in self.history]
        ), return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0])
        self.history.append({"role": "assistant", "content": response})
        return response
```

## AI Playground

Purpose: Interactive web UI for experimenting with models

```python
import gradio as gr
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")

def analyze_text(text, task):
    if task == "Sentiment":
        return sentiment(text)
    return "Unknown task"

gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(label="Input Text", lines=5),
        gr.Dropdown(["Sentiment"], label="Task")
    ],
    outputs="text",
    title="AI Playground"
).launch()
```

## Data Pipeline

Purpose: Process datasets for ML training/inference

```python
from datasets import load_dataset
from transformers import AutoTokenizer

class DataPipeline:
    def __init__(self, dataset_name):
        self.dataset = load_dataset(dataset_name)
        self.tokenizer = None
    
    def set_tokenizer(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def prepare(self, model_id, split="train"):
        self.set_tokenizer(model_id)
        return self.dataset[split]
```

## Fine-tuning Project

Purpose: Fine-tune models on custom datasets

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("imdb")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
trainer.train()
```

## Image Generator

Purpose: Create images from text prompts

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

def generate_image(prompt):
    return pipe(prompt, num_inference_steps=25).images[0]

image = generate_image("A beautiful sunset")
image.save("output.png")
```

## Summarizer

Purpose: Summarize documents quickly

```python
from transformers import pipeline
summarizer = pipeline("summarization")

def summarize_text(text):
    return summarizer(text, max_length=150)[0]["summary_text"]
```

## Quick Start Template

```python
"""
Project Name: [Your Project]
"""
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = "/content/.hf_cache"

from transformers import AutoModel, AutoTokenizer

MODEL_ID = "your-model-id"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    return model, tokenizer

def process(input_data):
    pass

if __name__ == "__main__":
    model, tokenizer = load_model()
```

All templates are MIT licensed. Use freely!