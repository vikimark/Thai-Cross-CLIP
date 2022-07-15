from transformers import (
    CamembertModel,
    CamembertTokenizer,
    CamembertConfig,
)
import torch
from torch import nn
from .config import CFG

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = CamembertModel.from_pretrained(model_name)
        else:
            self.model = CamembertModel(config=CamembertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TextModel(nn.Module):
    def __init__(
        self,
        text_embedding = CFG.text_embedding
    ):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.tokenizer = CamembertTokenizer.from_pretrained(CFG.text_tokenizer)

    def forward(self, batch):
        # Getting Text Features
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        # Project to the same dim of image encoder
        text_embeddings = self.text_projection(text_features)

        return text_embeddings
    
    def encode_text(self, text):
        tokened_word = self.tokenizer(text, padding=True, truncation=True, max_length=CFG.max_length)
        text_features = self.text_encoder(
            input_ids=torch.tensor(tokened_word["input_ids"]).to(CFG.device),
            attention_mask=torch.tensor(tokened_word["attention_mask"]).to(CFG.device)
        )
        text_embeddings = self.text_projection(text_features)
        return text_embeddings