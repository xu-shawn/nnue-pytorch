"""Exercise pooled native batches, training loss/backprop, and serialization."""
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path.cwd()))
import torch
import data_loader
import model as M
from model.modules.features import DEFAULT_FEATURES

def main():
    torch.set_num_threads(2)
    torch.manual_seed(7215)
    net = M.NNUE(M.NNUEConfig(features=DEFAULT_FEATURES, model_config=M.ModelConfig(L1=32)), max_epoch=1, num_batches_per_epoch=24)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".pgo/small.binpack")
    provider = data_loader.SparseBatchProvider(net.model.input_feature_name, [str(data)], 16, num_workers=2, device="cpu")
    for step in range(24):
        batch = next(provider)
        optimizer.zero_grad(set_to_none=True)
        loss = net.compute_loss(batch, step)
        assert torch.isfinite(loss), loss
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().max() > 0 for p in net.parameters())
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optimizer.step()
        net.model.clip_weights(include_input=True)
    del provider
    writer = M.NNUEWriter(net.model, ft_compression="leb128", verbose=False)
    with tempfile.TemporaryFile() as serialized:
        serialized.write(writer.buf)
        serialized.seek(0)
        loaded = M.NNUEReader(serialized, DEFAULT_FEATURES, net.config.model_config).model
    with torch.no_grad():
        args = [batch[0], batch[1], batch[2], batch[3], *batch[6:]]
        before = net.model(*args, fake_quantize_acts=True, fake_quantize_weights=True)
        after = loaded(*args, fake_quantize_acts=True, fake_quantize_weights=True)
        assert torch.allclose(before, after, rtol=0, atol=1e-6), (before-after).abs().max()
    print(f"PASS: 24 pooled batches / 384 positions, forward/backward/optimizer, export/reload; loss={loss.item():.8f}")


if __name__ == "__main__":
    main()
