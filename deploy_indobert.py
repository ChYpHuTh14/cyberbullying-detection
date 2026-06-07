import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import re


st.set_page_config(
    page_title="IndoBERT CyberBullying Analysis",
    layout="centered"
)


HF_MODEL = "ChYpHuTh14/indobert-cyberbullying"


labels = [
    "Cyberbullying",
    "Not Cyberbullying"
]

device=torch.device('cpu')

def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL
    )

    return tokenizer, model

tokenizer, model = load_model()
print(model)
print(tokenizer)
class CyberbullyingDataset(Dataset):
    def __init__(self,comments, labels,tokenizer,max_len=64):
        self.comments=comments
        self.labels=labels
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        encoding=self.tokenizer(
            self.comments[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids':encoding['input_ids'].squeeze(0),
            'attention_mask':encoding['attention_mask'].squeeze(0),
            'labels':torch.tensor(self.labels[idx],dtype=torch.long)
        }


slang=pd.read_csv('colloquial-indonesian-lexicon.csv')
slang_dict=dict(zip(slang['slang'],slang['formal']))

def replace_slang(comment):
    for word, replacement in slang_dict.items():
        comment = re.sub(r'\b' + re.escape(word) + r'\b', replacement, comment)
    return comment

#intensifier normalization
def normalize_intensifier(comment):
    return re.sub(r'(.)\1{2,}', r'\1\1', comment)

def preprocess_comment(comment):
    if not isinstance(comment, str):
        return ""
    # hapus emoji & karakter aneh
    comment=comment.encode("ascii", "ignore").decode("ascii")
    # lowercase
    comment=comment.lower()
    
    # hapus spasi berlebih
    comment=re.sub(r"\s+", " ", comment).strip()
    comment=re.sub(r"@\w+", "USER", comment)
    comment=normalize_intensifier(comment)
    comment=replace_slang(comment)
    
    return comment
def predict_sentiment(text):
    
    input_comment=[preprocess_comment(text)]
    input_comment=CyberbullyingDataset(input_comment,[0],tokenizer)
    input_ids = input_comment[0]["input_ids"].unsqueeze(0)
    attention_mask = input_comment[0]["attention_mask"].unsqueeze(0)

    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probabilities = F.softmax(
            logits,
            dim=1
        )
    pred=torch.argmax(logits,dim=1)
    probs = probabilities.squeeze().cpu().tolist()

    confidence = probs[pred]

    return pred, confidence, probs



st.title("IndoBERT Cyberbullying Prediction")

st.write(
    "Prediksi sentimen Cyberbullying atau Not Cyberbullying menggunakan IndoBERT."
)


text_input = st.text_area(
    "Input Text",
    placeholder="Contoh: Pelayanan restoran ini sangat memuaskan!"
)

if st.button("Predict"):

    if text_input.strip() == "":

        st.warning("Masukkan teks terlebih dahulu.")

    else:

        pred, confidence, probs = predict_sentiment(text_input)

        st.subheader("Prediction Result")

        st.write("Prediction:", labels[pred])

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )
        fig, ax = plt.subplots(figsize=(5,5))

        ax.pie(
            probs,
            labels=labels,
            autopct='%1.1f%%',
            explode=(0.05, 0),
            shadow=True,
            startangle=90
        )

        ax.set_title("Prediction Confidence")
        ax.axis('equal')
        st.pyplot(fig)
        
