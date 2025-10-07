A Rust implementation of Llama 2 inference engine.

The code is inspired by https://github.com/karpathy/llama2.c and https://github.com/leo-du/llama2.rs


## Instruction

First download the model:

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

and then run:

```bash
cargo run stories15M.bin 0.0 256 "Once upon a time"
```
