from random import randint
from string import printable
import numpy as np
import torch
from rapidfuzz.distance.Levenshtein import distance as ldistance
from torch.optim import AdamW

from ..models import EditDistanceModel


def pad_with_null(string: str, target_length: int):
    null_char = "\0"
    padding_needed = max(0, target_length - len(string))
    return (string + (null_char * padding_needed))[:target_length]


def string_to_tensor(string: str, length: int) -> torch.Tensor:
    """Converts a string to a tensor of character indices."""
    padded = pad_with_null(string, length)
    # Use ord() to get integer representation, clamp to vocab size
    indices = [min(ord(c), 127) for c in padded]
    return torch.tensor(indices, dtype=torch.long)


def random_char() -> str:
    pos = randint(0, len(printable) - 1)
    return printable[pos]


def random_str(length: int) -> str:
    return "".join([random_char() for _ in range(length)])


def mangle_string(source: str, d: int) -> str:
    """
    Efficiently mangles a string to approximately the target distance
    Uses list operations for better performance
    """
    if d <= 0:
        return source

    mangled = list(source)
    edits_made = 0
    max_attempts = d * 3  # Prevent infinite loops
    attempts = 0

    while edits_made < d and attempts < max_attempts:
        attempts += 1

        if len(mangled) == 0:
            position = 0
            edit = "insert"
        else:
            position = randint(0, len(mangled) - 1)
            edit = ["insert", "delete", "modify"][randint(0, 2)]

        if edit == "insert":
            mangled.insert(position, random_char())
            edits_made += 1
        elif edit == "modify" and len(mangled) > 0:
            old_char = mangled[position]
            new_char = random_char()
            if old_char != new_char:  # Only count as edit if actually different
                mangled[position] = new_char
                edits_made += 1
        elif edit == "delete" and len(mangled) > 0:
            mangled.pop(position)
            edits_made += 1

    return "".join(mangled)


def get_random_edit_distance(
    minimum: int, maximum: int, mean: float, dev: float
) -> int:
    sample = np.random.normal(loc=mean, scale=dev)
    sample = int(sample)
    return min(max(sample, minimum), maximum)


def get_homologous_pair(
    source: str, length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Use more reasonable distance distribution
    distance = get_random_edit_distance(1, min(length//4, 10), 3, 2)
    mangled = mangle_string(source, distance)

    # Verify actual distance and use it for training
    actual_distance = ldistance(source, mangled)

    return (
        string_to_tensor(source, length),
        string_to_tensor(mangled, length),
        torch.tensor(float(actual_distance), dtype=torch.float),
    )


def get_non_homologous_pair(
    length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = random_str(length)
    other = random_str(length)

    # Ensure strings are actually different
    max_attempts = 5
    attempt = 0
    while source == other and attempt < max_attempts:
        other = random_str(length)
        attempt += 1

    distance = ldistance(source, other)

    return (
        string_to_tensor(source, length),
        string_to_tensor(other, length),
        torch.tensor(float(distance), dtype=torch.float),
    )


def squared_euclidean_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    return torch.sum((v1 - v2) ** 2, dim=1)


def get_batch(
    size: int, batch_size: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    half_b = int(batch_size / 2)

    # Generate diverse source strings for homologous pairs
    h_pairs = []
    for _ in range(half_b):
        source = random_str(size)
        h_pairs.append(get_homologous_pair(source, size))

    non_h_pairs = [get_non_homologous_pair(size) for _ in range(half_b)]

    # Shuffle the batch to prevent learning batch order patterns
    all_pairs = h_pairs + non_h_pairs
    np.random.shuffle(all_pairs)

    return all_pairs


def estimate_M(length: int, num_samples: int = 1000) -> float:
    """Estimates M, the average Levenshtein distance for non-homologous pairs."""
    total_distance = 0.0
    for _ in range(num_samples):
        _, _, dist_tensor = get_non_homologous_pair(length)
        total_distance += dist_tensor.item()
    return total_distance / num_samples


def get_distances(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    distance_metric: str = "euclidean",
    M: float | None = None,
    embedding_dim: int | None = None
):
    first: torch.Tensor = torch.stack([b[0] for b in batch])
    first = model(first)

    second: torch.Tensor = torch.stack([b[1] for b in batch])
    second = model(second)

    ds = torch.stack([b[2] for b in batch])

    d_hats = squared_euclidean_distance(first, second)

    if M is not None and embedding_dim is not None:
        # r(n) = sqrt(M / (2n)) from paper Eq. 6
        # We need r(n)^2 * d_hats, so (M / (2n)) * d_hats
        scaling_factor_squared = M / (2 * embedding_dim)
        d_hats = d_hats * scaling_factor_squared

    return (d_hats, ds)


def approximation_error(d_hat: torch.Tensor, d: torch.Tensor):
    return torch.mean(torch.abs(d - d_hat))


def get_loss(d_hat: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """
    Wei et al. Poisson regression loss function
    """
    # Wei et al. Poisson regression with improved numerical stability
    # PNLL(d̂, d) = d̂ - d * ln(d̂) with better handling of edge cases
    epsilon = 1e-8
    d_hat_stable = torch.clamp(d_hat, min=epsilon)
    return torch.mean(d_hat_stable - d * torch.log(d_hat_stable))


def validate_training_data(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict:
    """Validate and analyze training batch quality"""
    distances = [b[2].item() for b in batch]

    stats = {
        'min_distance': min(distances),
        'max_distance': max(distances),
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'zero_distance_count': sum(1 for d in distances if d == 0),
        'high_distance_count': sum(1 for d in distances if d > 15)
    }

    return stats


def run_experiment(
    embedding_dim: int,
    model: torch.nn.Module,
    learning_rate: float,
    num_steps: int,
    size: int,
    batch_size: int,
    use_gradient_clipping: bool = True,
    max_grad_norm: float = 1.0,
    distance_metric: str = "euclidean"
):
    """
    Runs a training experiment with the given parameters and improved loss functions.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    final_loss = 0.0
    final_approx_error = 0.0

    # Estimate M once at the beginning of the experiment
    M_estimate = estimate_M(size)
    print(f"Estimated M (average non-homologous distance): {M_estimate:.2f}")

    for x in range(num_steps):
        batch = get_batch(size, batch_size)

        distances = get_distances(
            batch, model, distance_metric, M=M_estimate, embedding_dim=embedding_dim
        )
        loss = get_loss(distances[0], distances[1])

        if x % 10 == 0:
            print(
                f"step: {x}, loss: {loss.item()}, approx_error: {approximation_error(distances[0], distances[1]).item()}"
            )

        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss = loss.item()
        final_approx_error = approximation_error(distances[0], distances[1]).item()

    return final_loss, final_approx_error
