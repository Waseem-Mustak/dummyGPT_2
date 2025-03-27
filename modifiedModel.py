from gptModel import *
from dataLoader import *
from loadWeight import *

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)
model = GPTModel(BASE_CONFIG)

# freeze the weights of the GPT-2 model
for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
# add a linear layer for binary classification
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

# unfreeze the weights of the last transformer block
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
# unfreeze the weights of the final layer
for param in model.final_norm.parameters():
    param.requires_grad = True


# # //As i saved weight in model_and_optimizer.pth file, i will load the weight from that file so commenting this download and load part
# model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# from gpt_download3 import download_and_load_gpt2

# settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
# load_weights_into_gpt(model, params)


# # #/////////// LOAD WEIGHTS INTO MODEL
# checkpoint = torch.load("model_and_optimizer.pth")
# # model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# model.eval();
# text_1 = "Every effort moves you"

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids(text_1, tokenizer),
#     max_new_tokens=15,
#     context_size=BASE_CONFIG["context_length"]
# )

# print(token_ids_to_text(token_ids, tokenizer))

# # SAVE WEIGHTS INTO DEVICE
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     }, 
#     "model_and_optimizer.pth"
# )