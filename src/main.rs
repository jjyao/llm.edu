use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::f32;
use std::fs::File;
use std::io::{BufReader, Read, Write, stdout};
use std::sync::mpsc;
use std::thread;
use std::time::SystemTime;

trait FromBytes {
    fn from_bytes(bytes: [u8; 4]) -> Self;
}
impl FromBytes for u32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }
}
impl FromBytes for f32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

fn read<T: FromBytes>(rdr: &mut BufReader<File>) -> T {
    let mut buffer = [0u8; 4];
    rdr.read_exact(&mut buffer).expect("Error reading file");
    T::from_bytes(buffer)
}

fn read_vec<T: FromBytes>(rdr: &mut BufReader<File>, size: usize) -> Vec<T> {
    (0..size).map(|_| read::<T>(rdr)).collect()
}

#[derive(Clone)]
struct Token {
    subword: Vec<u8>,
    score: f32,
}

#[derive(Clone)]
struct Tokenizer {
    vocab: Vec<Token>,
}

impl Tokenizer {
    fn from_file(file: &str, vocab_size: usize) -> Self {
        let mut rdr = BufReader::new(File::open(file).expect("Couldn't load tokenizer file"));
        read::<u32>(&mut rdr) as usize; // max token length

        let mut vocab: Vec<Token> = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let score = read::<f32>(&mut rdr);
            let len = read::<u32>(&mut rdr) as usize;
            let mut subword = vec![0_u8; len];
            rdr.read_exact(&mut subword).unwrap();
            vocab.push(Token { subword, score });
        }

        Self { vocab }
    }

    fn encode(&self, text: &[u8]) -> Vec<usize> {
        let mut tokens: Vec<usize> = Vec::new();

        for i in 0..text.len() {
            let byte = &text[i..i + 1];
            let (id, _) = self
                .vocab
                .iter()
                .enumerate()
                .find(|x| (*(*x).1).subword == byte)
                .expect("illegal character");
            tokens.push(id);
        }

        loop {
            let mut best = (-1e10_f32, (usize::MAX, usize::MAX)); // (score, (vocab index, tokens index))

            for i in 0..tokens.len() - 1 {
                let mut buffer: Vec<u8> = Vec::new();
                buffer.extend(&self.vocab[tokens[i]].subword);
                buffer.extend(&self.vocab[tokens[i + 1]].subword);
                for (vid, token) in self.vocab.iter().enumerate() {
                    if token.subword == buffer {
                        if token.score > best.0 {
                            best = (token.score, (vid, i));
                        }
                    }
                }
            }

            if best.1.0 == usize::MAX {
                break; // no more possible merges
            }

            // perform merge
            tokens[best.1.1] = best.1.0;
            tokens.remove(best.1.1 + 1);
        }

        tokens
    }

    fn decode(&self, token: usize) -> Vec<u8> {
        self.vocab[token].subword.clone()
    }
}

#[derive(Clone, Debug)]
struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    max_seq_len: usize,
}

impl Config {
    fn from_buf_reader(f: &mut BufReader<File>) -> Self {
        Self {
            dim: read::<u32>(f) as usize,
            hidden_dim: read::<u32>(f) as usize,
            n_layers: read::<u32>(f) as usize,
            n_heads: read::<u32>(f) as usize,
            n_kv_heads: read::<u32>(f) as usize,
            vocab_size: read::<u32>(f) as usize,
            max_seq_len: read::<u32>(f) as usize,
        }
    }
}

#[derive(Clone)]
struct Weights {
    token_embedding_table: Vec<f32>,

    // weights for pre-attention rmsnorms
    rms_att_weight: Vec<f32>,

    // weights for kqv
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,

    // weights for pre-ffn rmsnorms
    rms_ffn_weight: Vec<f32>,

    // weights for ffn
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,

    // weights for final rmsnorm
    rms_final_weight: Vec<f32>,

    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<f32>,
    freq_cis_imag: Vec<f32>,
}

impl Weights {
    fn from_buf_reader(rdr: &mut BufReader<File>, config: &Config) -> Self {
        let head_size = config.dim / config.n_heads;
        Self {
            token_embedding_table: read_vec::<f32>(rdr, config.vocab_size * config.dim), // (vocab_size, dim)
            rms_att_weight: read_vec::<f32>(rdr, config.n_layers * config.dim), // (layer, dim)
            wq: read_vec::<f32>(rdr, config.n_layers * config.dim * config.dim), // (layer, n_heads * head_size, dim)
            wk: read_vec::<f32>(rdr, config.n_layers * config.dim * config.dim), // (layer, n_heads * head_size, dim)
            wv: read_vec::<f32>(rdr, config.n_layers * config.dim * config.dim), // (layer, n_heads * head_size, dim)
            wo: read_vec::<f32>(rdr, config.n_layers * config.dim * config.dim), // (layer, dim, n_heads * head_size)
            rms_ffn_weight: read_vec::<f32>(rdr, config.n_layers * config.dim),  // (layer, dim)
            w1: read_vec::<f32>(rdr, config.n_layers * config.hidden_dim * config.dim), // (layer, hidden_dim, dim)
            w2: read_vec::<f32>(rdr, config.n_layers * config.dim * config.hidden_dim), // (layer, dim, hidden_dim)
            w3: read_vec::<f32>(rdr, config.n_layers * config.hidden_dim * config.dim), // (layer, hidden_dim, dim)
            rms_final_weight: read_vec::<f32>(rdr, config.dim),                         // (dim,)
            freq_cis_real: read_vec::<f32>(rdr, config.max_seq_len * head_size / 2), // (max_seq_len, head_size/2)
            freq_cis_imag: read_vec::<f32>(rdr, config.max_seq_len * head_size / 2), // (max_seq_len, head_size/2)
        }
    }

    /// Create a shard of weights used by a single tensor parallel worker
    fn shard(&self, config: &Config, tp_size: usize, rank: usize) -> Self {
        let head_size = config.dim / config.n_heads;
        let n_heads_per_shard = config.n_heads / tp_size;
        let hidden_dim_per_shard = config.hidden_dim / tp_size;
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        let mut w1 = Vec::new();
        let mut w2 = Vec::new();
        let mut w3 = Vec::new();
        for layer in 0..config.n_layers {
            let layer_offset = layer * config.dim * config.dim;
            wq.extend(
                &self.wq[layer_offset + rank * n_heads_per_shard * head_size * config.dim
                    ..layer_offset + (rank + 1) * n_heads_per_shard * head_size * config.dim],
            );
            wk.extend(
                &self.wk[layer_offset + rank * n_heads_per_shard * head_size * config.dim
                    ..layer_offset + (rank + 1) * n_heads_per_shard * head_size * config.dim],
            );
            wv.extend(
                &self.wv[layer_offset + rank * n_heads_per_shard * head_size * config.dim
                    ..layer_offset + (rank + 1) * n_heads_per_shard * head_size * config.dim],
            );
            for r in 0..config.dim {
                let row_offset = layer_offset + r * config.dim;
                wo.extend(
                    &self.wo[row_offset + rank * n_heads_per_shard * head_size
                        ..row_offset + (rank + 1) * n_heads_per_shard * head_size],
                );
            }

            let layer_offset = layer * config.dim * config.hidden_dim;
            w1.extend(
                &self.w1[layer_offset + rank * hidden_dim_per_shard * config.dim
                    ..layer_offset + (rank + 1) * hidden_dim_per_shard * config.dim],
            );
            for r in 0..config.dim {
                let row_offset = layer_offset + r * config.hidden_dim;
                w2.extend(
                    &self.w2[row_offset + rank * hidden_dim_per_shard
                        ..row_offset + (rank + 1) * hidden_dim_per_shard],
                );
            }
            w3.extend(
                &self.w3[layer_offset + rank * hidden_dim_per_shard * config.dim
                    ..layer_offset + (rank + 1) * hidden_dim_per_shard * config.dim],
            );
        }
        Self {
            token_embedding_table: self.token_embedding_table.clone(),
            rms_att_weight: self.rms_att_weight.clone(),
            wq: wq, // (layer, n_heads_per_shard * head_size, dim)
            wk: wk, // (layer, n_heads_per_shard * head_size, dim)
            wv: wv, // (layer, n_heads_per_shard * head_size, dim)
            wo: wo, // (layer, dim, n_heads_per_shard * head_size)
            rms_ffn_weight: self.rms_ffn_weight.clone(),
            w1: w1, // (layer, hidden_dim_per_shard, dim)
            w2: w2, // (layer, dim, hidden_dim_per_shard)
            w3: w3, // (layer, hidden_dim_per_shard, dim)
            rms_final_weight: self.rms_final_weight.clone(),
            freq_cis_real: self.freq_cis_real.clone(),
            freq_cis_imag: self.freq_cis_imag.clone(),
        }
    }
}

struct KVCache {
    key_cache: Vec<f32>,   // (layer, max_seq_len, n_heads_per_shard * head_size)
    value_cache: Vec<f32>, // (layer, max_seq_len, n_heads_per_shard * head_size)
}

impl KVCache {
    fn new(config: &Config, tp_size: usize) -> Self {
        let head_size = config.dim / config.n_heads;
        let n_heads_per_shard = config.n_heads / tp_size;
        Self {
            key_cache: vec![
                0.0;
                config.n_layers * config.max_seq_len * n_heads_per_shard * head_size
            ],
            value_cache: vec![
                0.0;
                config.n_layers * config.max_seq_len * n_heads_per_shard * head_size
            ],
        }
    }
}

/// Apply RMSNorm for each row of a 2D tensor x (n, weight.len())
fn rmsnorm(x: &Vec<f32>, weight: &[f32]) -> Vec<f32> {
    let mut o = vec![0.0; x.len()];
    for r in 0..(x.len() / weight.len()) {
        let x = &x[r * weight.len()..(r + 1) * weight.len()];
        let o = &mut o[r * weight.len()..(r + 1) * weight.len()];

        // mean sum of squares
        let mss: f32 = x.iter().map(|&y| y * y).sum::<f32>() / (x.len() as f32);
        let rsqrt: f32 = 1.0 / (mss + 1e-5f32).sqrt();
        for ((oi, xi), wi) in o.iter_mut().zip(&x[..]).zip(weight) {
            *oi = *wi * rsqrt * *xi;
        }
    }

    o
}

fn matmul(x: &Vec<f32>, w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    // x (m, k) @ w (n, k)^T -> (m, n)
    let parallelism = 4;
    assert!(n % parallelism == 0);
    let mut o = vec![0.0; m * n];
    let o_addr = o.as_mut_ptr() as usize;
    thread::scope(|s| {
        for t in 0..parallelism {
            // each thread calculates `parallelism` number of columns of the output matrix
            s.spawn(move || {
                let o_ptr = o_addr as *mut f32;
                for r in 0..m {
                    for c in t * (n / parallelism)..(t + 1) * (n / parallelism) {
                        let mut val: f32 = 0.0;
                        for i in 0..k {
                            val += x[r * k + i] * w[c * k + i];
                        }
                        unsafe {
                            o_ptr.add(r * n + c).write(val);
                        }
                    }
                }
            });
        }
    });

    o
}

fn softmax(x: &mut [f32]) {
    let max: f32 = x.iter().fold(x[0], |a, &b| a.max(b));
    x.iter_mut().for_each(|a| *a = (*a - max).exp());
    let sum = x.iter().sum::<f32>();
    x.iter_mut().for_each(|a| *a /= sum);
}

#[derive(Clone)]
struct Model {
    config: Config,
    weights: Weights,
}

impl Model {
    fn from_checkpoint(ckpt_file: &str) -> Self {
        let mut rdr = BufReader::new(File::open(ckpt_file).expect("Couldn't load checkpoint file"));
        let config = Config::from_buf_reader(&mut rdr);
        let weights = Weights::from_buf_reader(&mut rdr, &config);
        Self { config, weights }
    }

    fn shard(&self, tp_size: usize, rank: usize) -> Self {
        Self {
            config: self.config.clone(),
            weights: self.weights.shard(&self.config, tp_size, rank),
        }
    }
}

enum Stage {
    Prefill,
    Decode,
}

/// https://danieldk.eu/Tensor-Parallelism
struct Worker {
    shard: Model,
    tokenizer: Tokenizer,
    tp_size: usize,

    rank: usize,
    tx: mpsc::Sender<Vec<f32>>,
    rx: mpsc::Receiver<Vec<f32>>,
}

impl Worker {
    fn run(&self, prompt_tokens: Vec<usize>, seed: u64, temperature: f32, steps: usize) {
        if self.rank == 0 {
            for token in prompt_tokens.iter() {
                print!(
                    "{}",
                    String::from_utf8(self.tokenizer.decode(*token)).unwrap()
                );
            }
            stdout().flush().unwrap();
        }

        let mut kv_cache = KVCache::new(&self.shard.config, self.tp_size);

        // prefill
        if prompt_tokens.len() > 1 {
            self.inference(
                prompt_tokens[0..prompt_tokens.len() - 1].to_vec(),
                0,
                &mut kv_cache,
                Stage::Prefill,
            );
        }

        // decode
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut pos: usize = prompt_tokens.len() - 1;
        let mut token: usize = prompt_tokens[prompt_tokens.len() - 1];
        while pos < steps {
            let logits = self.inference(vec![token], pos, &mut kv_cache, Stage::Decode);

            let next = self.sample(logits, &mut rng, temperature);

            if self.rank == 0 {
                print!(
                    "{}",
                    String::from_utf8(self.tokenizer.decode(next)).unwrap()
                );
                stdout().flush().unwrap();
            }

            token = next;
            pos += 1;
        }
    }

    /// https://medium.com/data-science/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da
    fn all_reduce(&self, mut data: Vec<f32>) -> Vec<f32> {
        let chunk_size = data.len() / self.tp_size;

        // share-reduce
        for step in 0..self.tp_size - 1 {
            let send_chunk_index = (self.rank + self.tp_size - step) % self.tp_size;
            let recv_chunk_index = (self.rank + self.tp_size - step - 1) % self.tp_size;

            // send to right
            self.tx
                .send(
                    data[send_chunk_index * chunk_size..(send_chunk_index + 1) * chunk_size]
                        .to_vec(),
                )
                .unwrap();

            // receive from left
            let recv_chunk = self.rx.recv().unwrap();
            for (i, val) in recv_chunk.iter().enumerate() {
                data[recv_chunk_index * chunk_size + i] += val;
            }
        }

        // share-only
        for step in 0..self.tp_size - 1 {
            let send_chunk_index = (self.rank + self.tp_size - step + 1) % self.tp_size;
            let recv_chunk_index = (self.rank + self.tp_size - step) % self.tp_size;

            // send to right
            self.tx
                .send(
                    data[send_chunk_index * chunk_size..(send_chunk_index + 1) * chunk_size]
                        .to_vec(),
                )
                .unwrap();

            // receive from left
            let recv_chunk = self.rx.recv().unwrap();
            data[recv_chunk_index * chunk_size..(recv_chunk_index + 1) * chunk_size]
                .clone_from_slice(&recv_chunk);
        }

        data
    }

    fn inference(
        &self,
        tokens: Vec<usize>,
        pos: usize, // pos of the first token in tokens
        kv_cache: &mut KVCache,
        stage: Stage,
    ) -> Vec<f32> {
        let config = &self.shard.config;
        let weights = &self.shard.weights;
        let head_size = config.dim / config.n_heads;
        let n_heads_per_shard = config.n_heads / self.tp_size;
        let hidden_dim_per_shard = config.hidden_dim / self.tp_size;

        // positional embedding
        let freq_cis_real_row =
            &weights.freq_cis_real[pos * (head_size / 2)..(pos + tokens.len()) * (head_size / 2)];
        let freq_cis_imag_row =
            &weights.freq_cis_imag[pos * (head_size / 2)..(pos + tokens.len()) * (head_size / 2)];

        // embed tokens into input vector x
        let mut x: Vec<f32> = Vec::new(); // (n, dim)
        for token in tokens.iter() {
            x.extend(
                &weights.token_embedding_table[token * (config.dim)..(token + 1) * (config.dim)],
            );
        }

        // run through layers
        for layer in 0..config.n_layers {
            // pre-attention norm
            let rms_normed_x = rmsnorm(
                &x,
                &weights.rms_att_weight[layer * (config.dim)..(layer + 1) * (config.dim)],
            );

            // qkv projection
            let mut query = matmul(
                &rms_normed_x,
                &weights.wq[layer * (n_heads_per_shard * head_size) * (config.dim)
                    ..(layer + 1) * (n_heads_per_shard * head_size) * (config.dim)],
                tokens.len(),
                config.dim,
                n_heads_per_shard * head_size,
            ); // rms_normed_x (n, dim) @ wq^T (dim, n_heads_per_shard * head_size) -> query (n, n_heads_per_shard * head_size)
            let mut key = matmul(
                &rms_normed_x,
                &weights.wk[layer * (n_heads_per_shard * head_size) * (config.dim)
                    ..(layer + 1) * (n_heads_per_shard * head_size) * (config.dim)],
                tokens.len(),
                config.dim,
                n_heads_per_shard * head_size,
            ); // rms_normed_x (n, dim) @ wk^T (dim, n_heads_per_shard * head_size) -> key (n, n_heads_per_shard * head_size)
            let value = matmul(
                &rms_normed_x,
                &weights.wv[layer * (n_heads_per_shard * head_size) * (config.dim)
                    ..(layer + 1) * (n_heads_per_shard * head_size) * (config.dim)],
                tokens.len(),
                config.dim,
                n_heads_per_shard * head_size,
            ); // rms_normed_x (n, dim) @ wv^T (dim, n_heads_per_shard * head_size) -> value (n, n_heads_per_shard * head_size)

            // rotary positional embedding
            for ti in 0..tokens.len() {
                for head in 0..n_heads_per_shard {
                    let token_offset = ti * (n_heads_per_shard * head_size);
                    let head_query = &mut query
                        [token_offset + head * head_size..token_offset + (head + 1) * head_size];
                    let head_key = &mut key
                        [token_offset + head * head_size..token_offset + (head + 1) * head_size];
                    for i in 0..(head_size / 2) {
                        let (fcr, fci) = (
                            freq_cis_real_row[ti * (head_size / 2) + i],
                            freq_cis_imag_row[ti * (head_size / 2) + i],
                        );
                        // rotate
                        (head_query[i * 2], head_query[i * 2 + 1]) = (
                            head_query[i * 2] * fcr - head_query[i * 2 + 1] * fci,
                            head_query[i * 2] * fci + head_query[i * 2 + 1] * fcr,
                        );
                        (head_key[i * 2], head_key[i * 2 + 1]) = (
                            head_key[i * 2] * fcr - head_key[i * 2 + 1] * fci,
                            head_key[i * 2] * fci + head_key[i * 2 + 1] * fcr,
                        );
                    }
                }
            }

            // cache kv values
            let layer_offset = layer * config.max_seq_len * n_heads_per_shard * head_size;
            kv_cache.key_cache[(layer_offset + pos * n_heads_per_shard * head_size)
                ..(layer_offset + (pos + tokens.len()) * n_heads_per_shard * head_size)]
                .copy_from_slice(&key);
            kv_cache.value_cache[(layer_offset + pos * n_heads_per_shard * head_size)
                ..(layer_offset + (pos + tokens.len()) * n_heads_per_shard * head_size)]
                .copy_from_slice(&value);

            let mut attention_output = vec![0.0; tokens.len() * n_heads_per_shard * head_size];
            // multihead attention
            for ti in 0..tokens.len() {
                for head in 0..n_heads_per_shard {
                    let token_query = &query[ti * n_heads_per_shard * head_size
                        ..(ti + 1) * n_heads_per_shard * head_size];
                    let head_query = &token_query[head * head_size..(head + 1) * head_size];
                    let mut scores = vec![0.0; pos + ti + 1];
                    for i in 0..(pos + ti + 1) {
                        let head_key_offset =
                            layer_offset + i * n_heads_per_shard * head_size + head * head_size;
                        let head_key =
                            &kv_cache.key_cache[head_key_offset..(head_key_offset + head_size)]; // key of the i-th position
                        // compute attention score
                        scores[i] = head_query
                            .iter()
                            .zip(head_key.iter()) // (head_query[i], head_key[i]) pairs
                            .map(|(&qi, &ki)| qi * ki)
                            .sum::<f32>()
                            / (head_size as f32).sqrt();
                    }

                    softmax(&mut scores);

                    let token_attention_output =
                        &mut attention_output[ti * n_heads_per_shard * head_size
                            ..(ti + 1) * n_heads_per_shard * head_size];
                    let head_attention_output =
                        &mut token_attention_output[head * head_size..(head + 1) * head_size];
                    // weighted sum of values
                    for i in 0..(pos + ti + 1) {
                        let head_value_offset =
                            layer_offset + i * n_heads_per_shard * head_size + head * head_size;
                        let head_value = &kv_cache.value_cache
                            [head_value_offset..(head_value_offset + head_size)]; // value of the i-th position
                        let score = scores[i];
                        head_attention_output
                            .iter_mut()
                            .zip(head_value)
                            .for_each(|(oi, &vi)| *oi += score * vi);
                    }
                }
            }
            // output projection
            let attention_output = matmul(
                &attention_output,
                &weights.wo[layer * (config.dim) * (n_heads_per_shard * head_size)
                    ..(layer + 1) * (config.dim) * (n_heads_per_shard * head_size)],
                tokens.len(),
                n_heads_per_shard * head_size,
                config.dim,
            ); // attention_output (n, n_heads_per_shard * head_size) @ wo^T (n_heads_per_shard * head_size, dim) -> attention_output (n, dim)

            let attention_output = self.all_reduce(attention_output);

            // residual connection -- add back to x
            x.iter_mut()
                .zip(attention_output.iter())
                .for_each(|(xi, oi)| *xi += *oi);

            // pre-ffn rmsnorm
            let rms_normed_x = rmsnorm(
                &x,
                &weights.rms_ffn_weight[layer * (config.dim)..(layer + 1) * (config.dim)],
            );

            // FFN block: self.w2(F.silu(self.w1(x)) * self.w3(x))
            let mut fnn1 = matmul(
                &rms_normed_x,
                &weights.w1[layer * hidden_dim_per_shard * (config.dim)
                    ..(layer + 1) * hidden_dim_per_shard * (config.dim)],
                tokens.len(),
                config.dim,
                hidden_dim_per_shard,
            ); // rms_normed_x (n, dim) @ w1^T (dim, hidden_dim_per_shard)  -> fnn1 (n, hidden_dim_per_shard)
            let fnn3 = matmul(
                &rms_normed_x,
                &weights.w3[layer * hidden_dim_per_shard * (config.dim)
                    ..(layer + 1) * hidden_dim_per_shard * (config.dim)],
                tokens.len(),
                config.dim,
                hidden_dim_per_shard,
            ); // rms_normed_x (n, dim) @ w3^T (dim, hidden_dim_per_shard) @  -> fnn3 (n, hidden_dim_per_shard)

            // apply silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            fnn1.iter_mut()
                .for_each(|a| *a = *a * (1.0 / (1.0 + (-*a).exp())));

            // elementwise multiply with hb2=w3(x) into hb
            fnn1.iter_mut().zip(fnn3.iter()).for_each(|(a, &b)| *a *= b);

            let fnn_output = matmul(
                &fnn1,
                &weights.w2[layer * (config.dim) * hidden_dim_per_shard
                    ..(layer + 1) * (config.dim) * hidden_dim_per_shard],
                tokens.len(),
                hidden_dim_per_shard,
                config.dim,
            ); // fnn1 (n, hidden_dim_per_shard) @ w2^T (hidden_dim_per_shard, dim) -> fnn_output (n, dim)

            let fnn_output = self.all_reduce(fnn_output);

            // residual connection
            x.iter_mut()
                .zip(fnn_output.iter())
                .for_each(|(xi, &oi)| *xi += oi);
        }

        if matches!(stage, Stage::Prefill) {
            return Vec::new();
        }

        // final rmsnorm
        let x = rmsnorm(&x, &weights.rms_final_weight);

        // compute logits
        let logits = matmul(
            &x,
            &weights.token_embedding_table,
            tokens.len(),
            config.dim,
            config.vocab_size,
        ); // x (n, dim) @ token_embedding_table^T (dim, vocab_size) @  -> logits (n, vocab_size)
        logits
    }

    fn sample(&self, mut logits: Vec<f32>, rng: &mut impl Rng, temperature: f32) -> usize {
        if temperature == 0.0 {
            // greedy decoding, choose argmax
            return logits
                .iter()
                .enumerate()
                .reduce(|(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) })
                .map(|(i, _)| i)
                .unwrap();
        }

        // temperature scaling
        if temperature < 1.0 {
            logits.iter_mut().for_each(|logit| *logit /= temperature);
        }
        // compute probabilities
        softmax(&mut logits);

        let r: f32 = rng.random();
        let mut cdf = 0.0;
        for (i, &p) in logits.iter().enumerate() {
            cdf += p;
            if r < cdf {
                return i;
            }
        }

        logits.len() - 1 // in case of rounding errors
    }
}

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the model checkpoint file
    #[arg(long)]
    checkpoint: String,

    #[arg(long)]
    temperature: f32,

    #[arg(long)]
    steps: usize,

    #[arg(long, default_value_t = String::new())]
    prompt: String,

    /// Tensor parallel size
    #[arg(long, default_value_t = 1)]
    tp_size: usize,

    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    // load model
    let model = Model::from_checkpoint(&args.checkpoint);
    assert!(
        model.config.n_kv_heads == model.config.n_heads,
        "MQA is not supported"
    );
    assert!(
        model.config.n_kv_heads % args.tp_size == 0,
        "n_kv_heads must be divisible by tp_size"
    );
    assert!(
        model.config.dim % args.tp_size == 0,
        "dim must be divisible by tp_size"
    );
    assert!(
        model.config.hidden_dim % args.tp_size == 0,
        "hidden_dim must be divisible by tp_size"
    );
    println!("Model {:?}", model.config);

    // load tokenizer
    let tokenizer = Tokenizer::from_file("tokenizer.bin", model.config.vocab_size);

    let mut prompt_tokens = match args.prompt.len() {
        0 => Vec::new(),
        _ => tokenizer.encode(args.prompt.as_bytes()),
    };
    println!("Prompt tokens: {:?}", prompt_tokens);
    prompt_tokens.insert(0, 1); // token 1 is <s> (bos) in the vocab

    // main generation loop
    let start_time = SystemTime::now();

    thread::scope(|s| {
        // create channels for ring all-reduce
        let mut txs = Vec::new();
        let mut rxs = Vec::new();
        for _ in 0..args.tp_size {
            let (tx, rx) = mpsc::channel::<Vec<f32>>();
            txs.push(Some(tx));
            rxs.push(Some(rx));
        }
        for rank in (0..args.tp_size).rev() {
            let worker = Worker {
                shard: model.shard(args.tp_size, rank),
                tokenizer: tokenizer.clone(),
                tp_size: args.tp_size,
                rank: rank,
                tx: txs[(rank + 1) % args.tp_size].take().unwrap(), // send to right
                rx: rxs[rank].take().unwrap(),                      // receive from left
            };
            if rank != 0 {
                s.spawn({
                    let worker = worker;
                    let prompt_tokens = prompt_tokens.clone();
                    move || worker.run(prompt_tokens, args.seed, args.temperature, args.steps)
                });
            } else {
                worker.run(
                    prompt_tokens.clone(),
                    args.seed,
                    args.temperature,
                    args.steps,
                );
            }
        }
    });

    let elapsed_time = start_time.elapsed().unwrap();
    println!();
    println!("--------------------------------");
    println!(
        "elapsed: {}.{:03} s, avg tok/s: {}",
        elapsed_time.as_secs(),
        elapsed_time.subsec_millis(),
        (args.steps - 1) as f32 / elapsed_time.as_secs_f32()
    );
}
