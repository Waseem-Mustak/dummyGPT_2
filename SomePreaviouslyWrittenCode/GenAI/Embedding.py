import torch,tiktoken
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    








def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


with open("tata.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()



# print("PyTorch version:", torch.__version__)
# dataloader = create_dataloader_v1(
#     raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
# )

# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)

# second_batch = next(data_iter)
# print(second_batch)


dataloader = create_dataloader_v1(raw_text, batch_size=3, max_length=4, stride=4, shuffle=False)

# token
all_data = list(dataloader)  # List of (inputs, targets) tuples
# print(all_data)  # First batch
for i, (inputs, targets) in enumerate(all_data):
    print(f"Batch {i}:\n")
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    print("\n\n")



# # input_embedding= token_em+pos_embedding
# vocab_size = 50257
# output_dimantion = 5
# max_length=4
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dimantion)
# position_embedding_layer = torch.nn.Embedding(max_length, output_dimantion)
# position_embedding = position_embedding_layer(torch.arange(max_length))

# # print(torch.arange(max_length))

# embeted_input_target = []
# for batch_idx, (inputs, targets) in enumerate(all_data):
#     token_embedding_input = token_embedding_layer(inputs)
#     # print(token_embedding_input)
#     token_embedding_target = token_embedding_layer(targets)
#     # print(token_embedding_target,"\n\n")
#     input_embedding = token_embedding_input + position_embedding
#     target_embedding = token_embedding_target + position_embedding
#     # print(inputs)
#     # print(input_embedding)
#     # print(targets)
#     # print(target_embedding)
#     # print("\n\n")
#     embeted_input_target.append([input_embedding,target_embedding])
# for (i,j) in embeted_input_target:
#     print("Input:")
#     print(i)
#     print("Target:")
#     print(j,"\n\n")

# # token_embedding = token_embedding_layer(all_data[0][1])
# # print(token_embedding)
# # position_embedding_layer = torch.nn.Embedding(max_length, output_dim)




# # data_iter = iter(dataloader)
# # # print(next(data_iter))
# # inputs, targets = next(data_iter)
# # # Tokenize the entire text
# # print("Inputs:\n", inputs)
# # print("\nTargets:\n", targets)

