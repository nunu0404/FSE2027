# E4 environment comparison

| Component | Original grid/battery | E4-A/E4-C | E4-B |
|---|---|---|---|
| Software environment | frozen Python/package snapshot | same | same |
| Inference engine | Transformers 5.9.0 | same | same |
| dtype/decoding | BF16, greedy, max 24 | same | same |
| prompt/pairs/images | SHA-pinned originals | same | same |
| physical GPU | GPU0 UUID GPU-98933e9f... | GPU0, same UUID | GPU1 UUID GPU-197872ce... |
| driver/CUDA | 580.95.05 / torch cu130 | same | same |

E4-B changes only the physical GPU UUID. Both devices are NVIDIA RTX PRO 6000
Blackwell Server Edition; software, engine, precision, and assets remain fixed.
