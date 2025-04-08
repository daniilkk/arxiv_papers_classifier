import streamlit as st
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertModel

from src import PaperClassifierV1, PaperClassifierDatasetV1


@st.cache_resource
def load_everything():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # DistilBertTokenizer.from_pretrained('distilbert-base-uncased') doesn't work from my laptop, but we don't need
    # that checkpoint anymore so we will use this class instead.
    class EmptyPaperClassifier(PaperClassifierV1):
        def __init__(self, n_classes):
            super(PaperClassifierV1, self).__init__()
            self.backbone = DistilBertModel(DistilBertConfig())
            self.head = torch.nn.Linear(in_features=self.backbone.config.hidden_size, out_features=n_classes)

    model = EmptyPaperClassifier(n_classes=len(PaperClassifierDatasetV1.MAJORS))
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer, device


def classify_paper(title, abstract, model, tokenizer, device):
    if abstract.strip() == "":
        inputs = tokenizer(
            title,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
    else:
        inputs = tokenizer(
            [title],
            [abstract],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    return pd.DataFrame({
        'Category': PaperClassifierDatasetV1.MAJORS,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)


def main(threshold: float = 0.5):
    st.set_page_config(page_title="ArXiv Paper Classifier", page_icon="🦈")
    st.title("ArXiv Paper Classifier")

    model, tokenizer, device = load_everything()

    col1, col2 = st.columns([1, 1])
    with col1:
        title = st.text_area("Title", height=200, placeholder="Enter paper title here...", ).strip()
    with col2:
        abstract = st.text_area("Abstract (optional)", height=200, placeholder="Enter paper abstract here...")

    if st.button("Classify", type='primary', use_container_width=True):
        if len(title) == 0:
            st.error("Please enter a paper title")
        else:
            with st.spinner('In progress...'):
                results = classify_paper(title, abstract, model, tokenizer, device)

            st.subheader("Results")

            predicted = results[results['Probability'] > threshold]['Category'].tolist()
            results['Probability'] = results['Probability'].apply(lambda x: f"{x:.2%}")

            if len(predicted) == 0:
                st.info("Hmm, I am not sure about this one.")
            else:
                st.success(f"Predicted categories: {', '.join(predicted)}")

            with st.expander("Show details"):
                st.dataframe(results, use_container_width=True, hide_index=True)
                st.caption("All categories with their confidence scores")

if __name__ == "__main__":
    main()
