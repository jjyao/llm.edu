use clap::Parser;
use fancy_regex::Regex;
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::time::SystemTime;

/// https://en.wikipedia.org/wiki/D-ary_heap
struct DaryMaxHeap<T: PartialOrd> {
    data: Vec<T>,
    dary: usize,
}

impl<T: PartialOrd> DaryMaxHeap<T> {
    fn new(dary: usize) -> Self {
        Self {
            data: Vec::new(),
            dary: dary,
        }
    }

    fn push(&mut self, value: T) {
        self.data.push(value);

        let mut current: usize = self.data.len() - 1;
        while current > 0 {
            let parent: usize = (current - 1) / self.dary;
            if self.data[current] > self.data[parent] {
                self.data.swap(current, parent);
                current = parent;
            } else {
                break;
            }
        }
    }

    fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }

        if self.data.len() == 1 {
            return self.data.pop();
        }

        let root = self.data.swap_remove(0);
        let mut current: usize = 0;
        while current <= (self.data.len() - 1 - 1) / self.dary {
            let mut max_child: usize = current * self.dary + 1;
            for i in 2..=self.dary {
                let child = current * self.dary + i;
                if child < self.data.len() && self.data[child] > self.data[max_child] {
                    max_child = child;
                }
            }
            if self.data[current] < self.data[max_child] {
                self.data.swap(current, max_child);
                current = max_child;
            } else {
                break;
            }
        }

        Some(root)
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

type Byte = u8;
type Token = usize;

struct Word {
    tokens: Vec<Token>,
    count: u64,
}

impl Word {
    fn new(word: &str, count: u64) -> Self {
        Self {
            tokens: word.as_bytes().iter().map(|b| *b as Token).collect(),
            count: count,
        }
    }
}

struct TokenPair {
    pair: (Token, Token),
    count: u64,                   // count of this token pair in the corpus
    word_indices: HashSet<usize>, // indices of words that contain this token pair
}

impl TokenPair {
    fn new(pair: (Token, Token)) -> Self {
        Self {
            pair: pair,
            count: 0,
            word_indices: HashSet::new(),
        }
    }
}

impl PartialEq for TokenPair {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl Eq for TokenPair {}

impl PartialOrd for TokenPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TokenPair {
    fn cmp(&self, other: &Self) -> Ordering {
        // max heap by count
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // ascending order on the pair when counts tie
            other.pair.cmp(&self.pair)
        }
    }
}

struct Tokenizer {
    vocab: HashMap<Token, Vec<Byte>>,
    merges: HashMap<(Token, Token), Token>,
}

impl Tokenizer {
    fn train(corpus: impl Iterator<Item = String>, vocab_size: usize) -> Self {
        // pre-tokenization
        let re =
            Regex::new(r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+")
                .unwrap(); // GPT-4 style regex pattern for splitting text
        let mut word_count_map: HashMap<String, u64> = HashMap::new();
        for text in corpus {
            for m in re.find_iter(&text) {
                let word = m.unwrap().as_str().to_string();
                word_count_map
                    .entry(word)
                    .and_modify(|c| *c += 1)
                    .or_insert(1);
            }
        }

        let mut words: Vec<Word> = Vec::new();
        for (word, count) in word_count_map.iter() {
            words.push(Word::new(word, *count));
        }

        let mut token_pairs: HashMap<(Token, Token), TokenPair> = HashMap::new();
        for (word_index, word) in words.iter().enumerate() {
            for i in 0..word.tokens.len() - 1 {
                let pair = (word.tokens[i], word.tokens[i + 1]);
                let token_pair = token_pairs.entry(pair).or_insert(TokenPair::new(pair));
                token_pair.count += word.count;
                token_pair.word_indices.insert(word_index);
            }
        }

        let mut token_pair_heap: DaryMaxHeap<TokenPair> = DaryMaxHeap::new(3);
        let mut token_pair_count_map: HashMap<(Token, Token), u64> = HashMap::new();
        for (_, token_pair) in token_pairs.drain() {
            token_pair_count_map.insert(token_pair.pair, token_pair.count);
            token_pair_heap.push(token_pair);
        }

        let mut vocab: HashMap<Token, Vec<Byte>> = HashMap::new();
        for i in 0..256 {
            vocab.insert(i as Token, vec![i as Byte]);
        }
        let mut merges: HashMap<(Token, Token), Token> = HashMap::new();

        while token_pair_heap.len() > 0 && vocab.len() < vocab_size {
            let mut token_pair = token_pair_heap.pop().unwrap();
            let token_pair_count = token_pair_count_map
                .get(&token_pair.pair)
                .copied()
                .unwrap_or(0);
            if token_pair.count != token_pair_count {
                if token_pair_count > 0 {
                    token_pair.count = token_pair_count;
                    token_pair_heap.push(token_pair);
                }
                continue;
            }

            let new_token = vocab.len() as Token;
            merges.insert(token_pair.pair, new_token);
            vocab.insert(
                new_token,
                [
                    vocab.get(&token_pair.pair.0).unwrap().as_slice(),
                    vocab.get(&token_pair.pair.1).unwrap().as_slice(),
                ]
                .concat(),
            );

            let mut token_pair_count_delta_map: HashMap<(Token, Token), i64> = HashMap::new();
            let mut new_token_pair_word_indices_map: HashMap<(Token, Token), HashSet<usize>> =
                HashMap::new();
            for word_index in token_pair.word_indices.iter() {
                let word = &mut words[*word_index];
                let mut i = 0;
                while i < word.tokens.len() - 1 {
                    if word.tokens[i] != token_pair.pair.0
                        || word.tokens[i + 1] != token_pair.pair.1
                    {
                        i += 1;
                        continue;
                    }

                    let word_count = word.count as i64;

                    if i > 0 {
                        token_pair_count_delta_map
                            .entry((word.tokens[i - 1], word.tokens[i]))
                            .and_modify(|c| *c -= word_count)
                            .or_insert(-word_count);
                        token_pair_count_delta_map
                            .entry((word.tokens[i - 1], new_token))
                            .and_modify(|c| *c += word_count)
                            .or_insert(word_count);
                        new_token_pair_word_indices_map
                            .entry((word.tokens[i - 1], new_token))
                            .or_insert(HashSet::new())
                            .insert(*word_index);
                    }
                    if i + 2 < word.tokens.len() {
                        token_pair_count_delta_map
                            .entry((word.tokens[i + 1], word.tokens[i + 2]))
                            .and_modify(|c| *c -= word_count)
                            .or_insert(-word_count);
                        token_pair_count_delta_map
                            .entry((new_token, word.tokens[i + 2]))
                            .and_modify(|c| *c += word_count)
                            .or_insert(word_count);
                        new_token_pair_word_indices_map
                            .entry((new_token, word.tokens[i + 2]))
                            .or_insert(HashSet::new())
                            .insert(*word_index);
                    }
                    word.tokens[i] = new_token;
                    word.tokens.remove(i + 1);
                    i += 1;
                }
            }
            for (pair, delta) in token_pair_count_delta_map.iter() {
                if *delta > 0 {
                    // new token pair
                    let mut new_token_pair = TokenPair::new(*pair);
                    new_token_pair.count = *delta as u64;
                    new_token_pair.word_indices =
                        new_token_pair_word_indices_map.remove(pair).unwrap();
                    token_pair_count_map.insert(new_token_pair.pair, new_token_pair.count);
                    token_pair_heap.push(new_token_pair);
                } else if *delta < 0 {
                    // existing token pair
                    let new_count = token_pair_count_map.get(pair).unwrap() - delta.abs() as u64;
                    if new_count == 0 {
                        token_pair_count_map.remove(pair);
                    } else {
                        token_pair_count_map.insert(*pair, new_count);
                    }
                }
            }
            token_pair_count_map.remove(&token_pair.pair);
        }

        Tokenizer {
            vocab: vocab,
            merges: merges,
        }
    }

    /// Encode a string into a list of tokens
    fn encode(&self, text: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = text.as_bytes().iter().map(|b| *b as Token).collect();

        loop {
            let mut best = (usize::MAX, Token::MAX);

            for i in 0..tokens.len() - 1 {
                let token_pair = (tokens[i], tokens[i + 1]);
                if let Some(merged_token) = self.merges.get(&token_pair) {
                    if *merged_token < best.1 {
                        best = (i, *merged_token);
                    }
                }
            }

            if best.1 == Token::MAX {
                break; // no more possible merges
            }

            // perform merge
            tokens[best.0] = best.1;
            tokens.remove(best.0 + 1);
        }

        tokens
    }

    /// Decode a list of tokens into a string
    fn decode(&self, tokens: &[Token]) -> String {
        let mut bytes = Vec::new();
        for token in tokens {
            bytes.extend_from_slice(&self.vocab[token]);
        }
        String::from_utf8(bytes).unwrap()
    }
}

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the corpus file
    #[arg(long)]
    corpus: String,
}

fn main() {
    let args = Args::parse();

    let start_time = SystemTime::now();
    let mut corpus = String::new();
    File::open(args.corpus)
        .unwrap()
        .read_to_string(&mut corpus)
        .unwrap();
    let tokenizer = Tokenizer::train(vec![corpus].into_iter(), 500);
    let elapsed_time = start_time.elapsed().unwrap();
    println!();
    println!("--------------------------------");
    println!(
        "elapsed: {}.{:03} s",
        elapsed_time.as_secs(),
        elapsed_time.subsec_millis()
    );
    assert_eq!(tokenizer.decode(&tokenizer.encode("amp")), "amp");
}
