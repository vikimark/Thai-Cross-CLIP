import torch

class CFG:
    # captions_path = captions_path
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder_model = "wangchanberta-base-att-spm-uncased"
    text_embedding = 768
    text_tokenizer = "wangchanberta-base-att-spm-uncased"
    max_length = 200

    pretrained = False
    trainable = True
    temperature = 1.0

    num_projection_layers = 1
    projection_dim = 512 
    dropout = 0.1