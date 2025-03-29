from codecs import ascii_encode
import torch
import rapidfuzz

model = torch.nn.Sequential(
    torch.nn.Conv1d(8, 64, 3, 1, 1),
    torch.nn.AvgPool1d(2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(64, 64, 3, 1, 1),
    torch.nn.AvgPool1d(2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(64, 64, 3, 1, 1),
    torch.nn.AvgPool1d(2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(64, 64, 3, 1, 1),
    torch.nn.AvgPool1d(2),
    torch.nn.ReLU(),
    torch.nn.Conv1d(64, 64, 3, 1, 1),
    torch.nn.AvgPool1d(2),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(320, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 80),
    torch.nn.BatchNorm1d(80)
)

def pad_with_null(string: str, target_length: int):
    null_char = '\0'
    padding_needed = max(0, target_length - len(string))
    return string + (null_char * padding_needed)

def string_to_tensor(string: str) -> torch.Tensor:
    padded = pad_with_null(string, 160)
    b8s = [bin(b)[2:].zfill(8) for b in ascii_encode(padded)[0]]
    tensors = [torch.tensor([int(bit) for bit in b], dtype=torch.float) for b in b8s]
    t = torch.stack(tensors)
    return t.transpose(0,1)


t = string_to_tensor("unbelievable absolutely unbelievable!")
a = string_to_tensor("makekekeke ejeje")

tens = torch.stack([t,a])
print(tens.shape)

print(model(tens).shape)
