
from basics import *
from torch.utils.data import DataLoader, TensorDataset
from EncOnly import CausalTransformer
from EncDec import EncDecTransformer
from SoftAlignment import SoftAlignModel
import inspect

class FullModel(nn.Module):
    def __init__(self, model_type: str, input_dim=35, embed_dim=128):
        super().__init__()
        self.embedding = EmbedderGRNWrapper(input_dim=input_dim, embed_dim=embed_dim)

        if model_type == 'causal':
            self.model = CausalTransformer(input_dim=input_dim, embed_dim=embed_dim)
        elif model_type == 'encdec':
            self.model = EncDecTransformer(embed_dim=embed_dim, n_heads=4)
        elif model_type == 'softalign':
            self.model = SoftAlignModel(embed_dim=embed_dim)
        else:
            raise ValueError("Invalid model_type")

        # maps (B, T, embed_dim) → (B, T, input_dim)
        self.decoder = Decoder(embed_dim, input_dim)

    def forward(self,
                x: torch.Tensor,
                tgt: torch.Tensor = None,
                src_mask: torch.Tensor = None,
                src_padding_mask: torch.Tensor = None,
                max_len: int = 1
                ) -> torch.Tensor:
        x_embed = self.embedding(x)

        model_args = {
            "x": x_embed,
            "tgt": tgt,
            "src_mask": src_mask,
            "src_padding_mask": src_padding_mask,
            "max_len": max_len,
        }

        sig = inspect.signature(self.model.forward)
        filtered_args = {k: v for k, v in model_args.items() if k in sig.parameters and v is not None}

        out = self.model(**filtered_args)
        out = self.decoder(out)
        return out.squeeze(1)  # assuming output sequence length = 1


def train_full_model(
    model_type: str,
    data_dir: str,
    batch_size: int = 32,
    lr: float = 1e-3,
    epochs: int = 50,
    device: str = None,
    save_path: str = "best_model.pt"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_tensors, masks = get_data_and_mask(data_dir)
    x_train, y_train = data_tensors.train_input, data_tensors.train_target.squeeze(2)
    x_val, y_val = data_tensors.val_input, data_tensors.val_target.squeeze(2)
    train_mask = masks['train']
    val_mask = masks['val']

    print("Got data tensors:")
    print(data_tensors.summarize())

    train_ds = TensorDataset(x_train, y_train, train_mask)
    val_ds = TensorDataset(x_val, y_val, val_mask)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FullModel(model_type, input_dim=x_train.size(-1), embed_dim=128)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for (xb, yb, mb) in tqdm(train_loader, desc=f"Epoch {epoch:>2}"):
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)

            optimizer.zero_grad()
            src_padding_mask = (mb == 0)  # True where padding

            preds = model(xb, src_padding_mask=src_padding_mask)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for bidx, (xb, yb, mb) in enumerate(val_loader):
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                src_padding_mask = (mb == 0)

                preds = model(xb, src_padding_mask=src_padding_mask)
                total_val += criterion(preds, yb).item() * xb.size(0)

        avg_val = total_val / len(val_loader.dataset)
        print(f"Epoch {epoch:>2}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"  ↪️  8New best model saved (val loss {best_val_loss:.4f})")

    print("Training complete.")
    return model


if __name__ == "__main__":
    model = train_full_model(
        model_type="encdec",
        data_dir="Hayson/data",
        batch_size=32,
        lr=1e-3,
        epochs=30,
        save_path="best_encdec.pt",
        device="mps"
    )