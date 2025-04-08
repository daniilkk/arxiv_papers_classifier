import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from transformers import DistilBertTokenizer, DistilBertModel


class PaperClassifierDatasetV1(Dataset):
    MAJORS = ('cs', 'math', 'physics', 'q-bio', 'q-fin', 'stat', 'econ', 'eess')
    def __init__(self, csv_path: str, no_abstract_proba: float = 0., n_samples: int = 0):
        super().__init__()
        self.major_to_idx = {major : idx for idx, major in enumerate(self.MAJORS)}
        self.n_classes = len(self.MAJORS)

        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        self.df = pd.read_csv(csv_path)

        if n_samples == 0:
            n_samples = self.df.shape[0]

        self.x = self._tokenizer(
            list(zip(self.df['title'], self.df['abstract']))[:n_samples],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        self.y = torch.zeros((n_samples, len(self.MAJORS)))
        for row_idx, majors in enumerate(self.df['majors'][:n_samples]):
            majors = eval(majors)
            col_idxs = [self.major_to_idx[major] for major in majors]
            self.y[row_idx, col_idxs] = 1

        self.sep_token_id = self._tokenizer.sep_token_id
        self.sep_positions = list()
        for row_idx in range(len(self.x['input_ids'])):
            input_ids = self.x['input_ids'][row_idx]
            sep_pos = (input_ids == self.sep_token_id).nonzero(as_tuple=True)[0][0]
            self.sep_positions.append(sep_pos)

        self.no_abstract_proba = no_abstract_proba

    def __getitem__(self, index: int):
        input_ids = self.x['input_ids'][index, ...]
        attention_mask = self.x['attention_mask'][index, ...].clone()

        if torch.rand(1).item() < self.no_abstract_proba:
            sep_pos = self.sep_positions[index]
            attention_mask[sep_pos+1:] = 0

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target': self.y[index, ...]
        }

    def __len__(self):
        return self.x['input_ids'].shape[0]


class PaperClassifierV1(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.head = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        backbone_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        logits = self.head(backbone_output[:, 0, ...])
        return logits
