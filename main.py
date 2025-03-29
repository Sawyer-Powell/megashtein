from codecs import ascii_encode
from enum import Enum
from random import randint
from string import printable
import numpy as np
import torch
from rapidfuzz.distance.Levenshtein import distance as ldistance

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

def string_to_tensor(string: str, length: int) -> torch.Tensor:
    padded = pad_with_null(string, length)
    b8s = [bin(b)[2:].zfill(8) for b in ascii_encode(padded)[0]]
    tensors = [torch.tensor([int(bit) for bit in b], dtype=torch.float) for b in b8s]
    t = torch.stack(tensors)
    return t.transpose(0,1)

def random_char() -> str:
    pos = randint(0, len(printable)-1)
    return printable[pos]

def random_str(length: int) -> str:
    return "".join([random_char() for _ in range(length)])

def mangle(source: str, d: int) -> str:
    """
    Randomly edits a string until the desired levenshtein distance
    from the source string is reached
    """
    mangled = source

    class Edit(Enum):
        INSERT = 0
        DELETE = 1
        MODIFY = 2

    while ldistance(source, mangled) < d:
        if len(mangled) <= 1:
            position = 0
            edit = Edit.INSERT
        else:
            position = randint(0, len(mangled)-1)
            edit = Edit(randint(0, 2))

        if edit == Edit.INSERT:
            mangled = mangled[:position] + random_char() + mangled[position:]
        elif edit == Edit.MODIFY:
            mangled = mangled[:position] + random_char() + mangled[position+1:]
        elif edit == Edit.DELETE:
            mangled = mangled[:position] + mangled[position+1:]

    return mangled

def get_random_edit_distance(minimum: int, maximum: int, mean: float, dev: float) -> int:
    sample = np.random.normal(loc=mean, scale=dev)
    sample = int(sample)
    return min(max(sample, minimum), maximum)

def get_homologous_pair(source: str, length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    distance = get_random_edit_distance(0, length, 6, 4)
    mangled = mangle(source, distance)

    return (string_to_tensor(source, length), string_to_tensor(mangled, length), distance)

def get_non_homologous_pair(length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    source = random_str(length)
    other = random_str(length)
    distance = ldistance(source, other)

    return (string_to_tensor(source, length), string_to_tensor(other, length), distance)

print(get_non_homologous_pair(10))

"""
t = string_to_tensor("unbelievable absolutely unbelievable!")
a = string_to_tensor("makekekeke ejeje")

tens = torch.stack([t,a])
print(tens.shape)

print(model(tens).shape)
"""
