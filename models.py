import torch

class EditDistanceModel(torch.nn.Module):

    def __init__(self, vocab_size=128, embedding_dim=16, input_length=80):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(embedding_dim, 64, 3, 1, 1),
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
        )

        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_length, dtype=torch.long)
            dummy_embedded = self.embedding(dummy_input)
            dummy_permuted = dummy_embedded.permute(0, 2, 1)
            dummy_conved = self.conv_layers(dummy_permuted)
            flattened_size = self.flatten(dummy_conved).shape[1]

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 80),
            torch.nn.BatchNorm1d(80),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.BatchNorm1d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
