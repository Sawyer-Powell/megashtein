from codecs import ascii_encode
from enum import Enum
from random import randint
from string import printable
import numpy as np
import torch
from rapidfuzz.distance.Levenshtein import distance as ldistance
from torch.optim import AdamW

model: torch.nn.Sequential = torch.nn.Sequential(
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
    torch.nn.BatchNorm1d(80),
)


def pad_with_null(string: str, target_length: int):
    null_char = "\0"
    padding_needed = max(0, target_length - len(string))
    return (string + (null_char * padding_needed))[:target_length]


def string_to_tensor(string: str, length: int) -> torch.Tensor:
    padded = pad_with_null(string, length)
    b8s = [bin(b)[2:].zfill(8) for b in ascii_encode(padded)[0]]
    tensors = [torch.tensor([int(bit) for bit in b], dtype=torch.float) for b in b8s]
    t = torch.stack(tensors)
    return t.transpose(0, 1).requires_grad_(True)


def random_char() -> str:
    pos = randint(0, len(printable) - 1)
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
            position = randint(0, len(mangled) - 1)
            edit = Edit(randint(0, 2))

        if edit == Edit.INSERT:
            mangled = mangled[:position] + random_char() + mangled[position:]
        elif edit == Edit.MODIFY:
            mangled = mangled[:position] + random_char() + mangled[position + 1 :]
        elif edit == Edit.DELETE:
            mangled = mangled[:position] + mangled[position + 1 :]

    return mangled


def get_random_edit_distance(
    minimum: int, maximum: int, mean: float, dev: float
) -> int:
    sample = np.random.normal(loc=mean, scale=dev)
    sample = int(sample)
    return min(max(sample, minimum), maximum)


def get_homologous_pair(
    source: str, length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = get_random_edit_distance(1, length, 6, 4)
    mangled = mangle(source, distance)

    return (
        string_to_tensor(source, length),
        string_to_tensor(mangled, length),
        torch.tensor(distance, dtype=torch.float),
    )


def get_non_homologous_pair(
    length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = random_str(length)
    other = random_str(length)
    distance = ldistance(source, other)

    return (
        string_to_tensor(source, length),
        string_to_tensor(other, length),
        torch.tensor(distance, dtype=torch.float),
    )


def squared_euclidean_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    return torch.sum((v1 - v2) ** 2, dim=1)


def REchi_squared(d_hat: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    log2_e = 1.4426950408889634
    return (
        d / 2
        + (torch.lgamma(d / 2) / torch.log(torch.tensor(2.0)))
        - (d / 2 - 1) * torch.log2(d_hat)
        + d_hat / 2 * log2_e
    )


def get_batch(
    size: int, batch_size: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    half_b = int(batch_size / 2)
    source = random_str(size)
    h_pairs = [get_homologous_pair(source, size) for _ in range(half_b)]
    non_h_pairs = [get_non_homologous_pair(size) for _ in range(half_b)]

    return h_pairs + non_h_pairs


def get_distances(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model: torch.nn.Sequential,
):
    first: torch.Tensor = torch.stack([b[0] for b in batch])
    first = model(first)

    second: torch.Tensor = torch.stack([b[1] for b in batch])
    second = model(second)

    ds = torch.stack([b[2] for b in batch])
    d_hats = squared_euclidean_distance(first, second)

    return (d_hats, ds)


def approximation_error(d_hat: torch.Tensor, d: torch.Tensor):
    return torch.mean(torch.abs(d - d_hat))


def get_loss(d_hat: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return torch.mean(REchi_squared(d_hat, d))


SIZE = 160
BATCH_SIZE = 64

optimizer = AdamW(model.parameters(), lr=5e-6)

for x in range(1000):
    batch = get_batch(SIZE, BATCH_SIZE)
    distances = get_distances(batch, model)
    loss = get_loss(distances[0], distances[1])
    print(approximation_error(distances[0], distances[1]).item())
    loss.backward()
    optimizer.step()
