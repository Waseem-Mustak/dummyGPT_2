from gptModel import *
import tiktoken

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context

    ###Input batch:
 ###tensor([[6109, 3626, 6100,  345],
        ##[6109, 1110, 6622,  257]])
    
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond) ### batch, n_tokens, vocab_size
            # print(logits.shape)
        # print(idx_cond.shape)
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

# start_context = "Every effort moves you"


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



# def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
#                        eval_freq, eval_iter, start_context, tokenizer):
#     # Initialize lists to track losses and tokens seen
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen, global_step = 0, -1

#     # Main training loop
#     for epoch in range(num_epochs):
#         model.train()  # Set model to training mode
        
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad() # Reset loss gradients from previous batch iteration
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward() # Calculate loss gradients
#             optimizer.step() # Update model weights using loss gradients
#             tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
#             global_step += 1

#             # Optional evaluation step
#             if global_step % eval_freq == 0: 
#                 train_loss, val_loss = evaluate_model(
#                     model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): "
#                       f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

#         # Print a sample text after each epoch
#         generate_and_print_sample(
#             model, tokenizer, device, start_context
#         )

#     return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=100, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()




# Note:
# Uncomment the following code to calculate the execution time
import time
start_time = time.time()

torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")

checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
start_context="even through the prism"
generate_and_print_sample(
            model, tokenizer, device, start_context
        )

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Prompt completed in {execution_time_minutes:.2f} minutes.")


















# def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

#     # For-loop is the same as before: Get logits, and only focus on last time step
#     for _ in range(max_new_tokens):
#         idx_cond = idx[:, -context_size:]
#         with torch.no_grad():
#             logits = model(idx_cond)
#         logits = logits[:, -1, :]

#         # New: Filter logits with top_k sampling
#         if top_k is not None:
#             # Keep only top_k values
#             top_logits, _ = torch.topk(logits, top_k)
#             min_val = top_logits[:, -1]
#             logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

#         # New: Apply temperature scaling
#         if temperature > 0.0:
#             logits = logits / temperature

#             # Apply softmax to get probabilities
#             probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

#             # Sample from the distribution
#             idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

#         # Otherwise same as before: get idx of the vocab entry with the highest logits value
#         else:
#             idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

#         if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
#             break

#         # Same as before: append sampled index to the running sequence
#         idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

#     return idx


# torch.manual_seed(123)
# tokenizer = tiktoken.get_encoding("gpt2")

# checkpoint = torch.load("model_and_optimizer.pth")
# model = GPTModel(GPT_CONFIG_124M)
# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# token_ids = generate(
#     model=model,
#     idx=text_to_token_ids("I HAD always thought", tokenizer),
#     max_new_tokens=15,
#     context_size=GPT_CONFIG_124M["context_length"],
#     top_k=25,
#     temperature=1.4
# )
# for i in range (2):
#     print(token_ids_to_text(token_ids, tokenizer))
#     print("\n")
# # print("Output text:\n", token_ids_to_text(token_ids, tokenizer))