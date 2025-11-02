A Rust implementation of Llama 2 inference engine.

The code is inspired by https://github.com/karpathy/llama2.c and https://github.com/leo-du/llama2.rs.

## Instruction

First download the model:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

and then run:

```bash
cargo run --release -- --checkpoint stories15M.bin --temperature 0.0 --steps 256 --prompt "Once upon a time" --tp-size=2
```

## Optimizations

### KV Cache
Key and value tensors of previous tokens are cached.

### Prefill-Decode
Prefill and decode stages are separated.

### Tensor Parallelism
Each TP worker is one thread and all-reduce collective communication is implemented by message passing via channel.
