# HnM-RecSys
A pocket-size transformer trained to guess the *next* item that will land in a grocery basket.

---

## quick glance

| size | 1.2 M parameters |
| context | 50 items |
| vocab | 12 000 products |
| metric | MAP@12 |
| training wall-time | ≈ 90 min on M1 Pro (3 epochs) |
| hardware | Apple‑MPS / CUDA / CPU |

---

## architecture

decoder-only transformer  
GPTModel
├─ tok_emb        12 000 × 96\
├─ pos_emb        50 × 96\
├─ 2 × TransformerBlock\
│  ├─ MultiHeadAttention  6 heads, 96 dim\
│  └─ FeedForward         96 → 384 → 96\
└─ linear head → 12 000 logits\

---

## data flow

raw baskets → pad / truncate → map to 12 k most frequent ids → sliding window (input = t0 … tN-1, target = tN).

---

## training
python train.py

* 30 min/epoch, 3 epochs total  
* AdamW, lr = 3e-4, batch = 64  
* model + vocab mapping saved to `checkpoints/basket_transformer.pt`

---

## evaluation
python eval.py 

reports MAP@12 on 500 random test baskets plus distribution stats.

---

## file map
├── model.py            # GPTModel + blocks
├── train.py            # Dataset + training loop
├── evaluate.py         # MAP@12 evaluator
└── checkpoints/
└── basket_transformer.pt


---

## reproduce

1. place `train_padded.pt`, `val_padded.pt`, `test_padded.pt` under `data/splits/`.  
2. place `idx2product.json` and `product2idx.json` under `data/mappings/`.  
3. run `python train.py`, then `python evaluate.py`.

---

## notes

no external libraries beyond pytorch & tqdm.  
code is intentionally compact—tweak `CFG` at the top of `train.py` for larger runs.
