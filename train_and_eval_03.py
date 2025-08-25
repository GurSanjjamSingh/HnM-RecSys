# Basket Recommender Training - 1.5 hours total
import os, json, random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_03 import GPTModel

# === CONFIG - 30min/epoch, 3 epochs = 1.5 hours ===
CFG = dict(
    vocab_size     = 12000,   # Your requirement
    context_length = 50,      # Your requirement  
    emb_dim        = 96,      # Reduced for memory
    n_heads        = 6,       # emb_dim divisible by n_heads
    n_layers       = 2,       # Reduced layers
    drop_rate      = 0.1,
    qkv_bias       = False,
    batch_size     = 64,      # Reduced for MPS memory
    lr             = 3e-4,
    num_epochs     = 3,       # Your requirement
    device         = 'mps',
    grad_clip_norm = 1.0,
    weight_decay   = 0.01,
    seed           = 42,
    # Model params for your custom transformer
    use_dropout    = True,
    use_layer_norm = True,
    norm_first     = True,
    mixed_precision = False,
    compile_model  = False,
    optimizer      = 'adamw',
    betas          = (0.9, 0.95),
    eps            = 1e-8,
)

class BasketDataset(Dataset):
    def __init__(self, tensor_path, vocab_size_limit=None):
        print(f"Loading data from {tensor_path}...")
        
        sequences = torch.load(tensor_path, map_location='cpu')
        print(f"Loaded {len(sequences)} sequences, shape: {sequences.shape}")
        
        # Vocabulary mapping to limit size
        if vocab_size_limit:
            print(f"Mapping vocabulary to max {vocab_size_limit} items...")
            
            # Count item frequencies
            item_counts = {}
            for seq in sequences:
                for item in seq[seq != 0]:  # Skip padding
                    item_val = item.item()
                    item_counts[item_val] = item_counts.get(item_val, 0) + 1
            
            # Keep top frequent items
            top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
            top_items = top_items[:vocab_size_limit-1]  # Reserve 0 for padding
            
            # Create mapping: original_id -> new_id
            self.vocab_map = {0: 0}  # Keep padding
            for i, (item_id, count) in enumerate(top_items):
                self.vocab_map[item_id] = i + 1
            
            print(f"Vocabulary mapped: {len(item_counts)} -> {len(self.vocab_map)} items")
            self.actual_vocab_size = len(self.vocab_map)
        else:
            self.actual_vocab_size = int(sequences.max().item()) + 1
            self.vocab_map = None
        
        # Create input-target pairs
        self.inputs = []
        self.targets = []
        
        for seq in sequences:
            # Apply vocab mapping if needed
            if self.vocab_map:
                mapped_seq = torch.tensor([self.vocab_map.get(item.item(), 0) for item in seq])
            else:
                mapped_seq = seq
            
            # Find actual sequence (skip leading padding)
            nonzero_mask = mapped_seq != 0
            if nonzero_mask.sum() >= 2:  # Need at least 2 items
                nonzero_indices = nonzero_mask.nonzero().flatten()
                start_idx = nonzero_indices[0].item()
                actual_seq = mapped_seq[start_idx:]
                
                if len(actual_seq) >= 2:
                    # Create input-target pair
                    input_seq = actual_seq[:-1]  # All but last
                    target = actual_seq[-1].item()  # Last item
                    
                    # Pad/truncate input to context_length - 1
                    max_input_len = CFG['context_length'] - 1
                    if len(input_seq) > max_input_len:
                        input_seq = input_seq[-max_input_len:]  # Keep recent items
                    else:
                        # Pad with zeros at the beginning
                        pad_len = max_input_len - len(input_seq)
                        input_seq = torch.cat([torch.zeros(pad_len, dtype=torch.long), input_seq])
                    
                    self.inputs.append(input_seq)
                    self.targets.append(target)
        
        print(f"Created {len(self.inputs)} training examples")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], torch.tensor(self.targets[idx], dtype=torch.long)

def train():
    print("🛒 BASKET RECOMMENDATION TRAINING")
    print(f"Target: 30min/epoch × 3 epochs = 1.5 hours")
    print("💾 MPS Memory Optimized Version")
    print("=" * 50)
    
    # Clear MPS cache before starting
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("🧹 MPS cache cleared")
    
    # Load datasets
    train_ds = BasketDataset("data/splits/train_padded.pt", vocab_size_limit=CFG['vocab_size'])
    val_ds = BasketDataset("data/splits/val_padded.pt", vocab_size_limit=CFG['vocab_size'])
    
    # Update config with actual vocab size
    CFG['vocab_size'] = train_ds.actual_vocab_size
    print(f"Using vocab size: {CFG['vocab_size']}")
    
    # DataLoaders - smaller batch for MPS
    train_dl = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
    
    # Your custom transformer
    model = GPTModel(CFG).to(CFG['device'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    
    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CFG['lr'], 
        weight_decay=CFG['weight_decay'],
        betas=CFG['betas'], 
        eps=CFG['eps']
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"Training batches/epoch: {len(train_dl)}")
    print(f"Validation batches: {len(val_dl)}")
    print("=" * 50)
    
    best_val_loss = float('inf')
    
    for epoch in range(CFG['num_epochs']):
        print(f"\n📅 EPOCH {epoch+1}/{CFG['num_epochs']}")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_dl, desc=f"Training")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(CFG['device']), y.to(CFG['device'])
            
            # Forward pass
            logits = model(x)  # [batch, seq_len, vocab_size]
            logits = logits[:, -1, :]  # Last position for next-token prediction
            loss = loss_fn(logits, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip_norm'])
            optimizer.step()
            
            # Clear cache periodically to prevent memory buildup
            if batch_idx % 50 == 0:
                if CFG['device'] == 'mps':
                    torch.mps.empty_cache()
                elif CFG['device'] == 'cuda':
                    torch.cuda.empty_cache()
            
            # Metrics
            train_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)
            
            # Update progress bar
            if batch_idx % 20 == 0:
                current_acc = train_correct / train_total if train_total > 0 else 0
                avg_loss = train_loss / (batch_idx + 1)
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    avg_loss=f"{avg_loss:.3f}",
                    acc=f"{current_acc:.3f}"
                )
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in tqdm(val_dl, desc="Validating"):
                x, y = x.to(CFG['device']), y.to(CFG['device'])
                logits = model(x)[:, -1, :]
                loss = loss_fn(logits, y)
                
                val_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
        
        # Epoch results
        train_loss_avg = train_loss / len(train_dl)
        val_loss_avg = val_loss / len(val_dl)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"\nResults:")
        print(f"  Train: Loss={train_loss_avg:.4f}, Accuracy={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss_avg:.4f}, Accuracy={val_acc:.4f}")
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': CFG,
                'vocab_map': getattr(train_ds, 'vocab_map', None),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, "checkpoints/basket_transformer.pt")
            print(f"  ✅ Best model saved (val_loss={best_val_loss:.4f})")
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📊 Final Results:")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Model Size: {total_params:,} parameters")
    print(f"   Vocab Size: {CFG['vocab_size']:,}")
    print(f"📁 Model saved: checkpoints/basket_transformer.pt")
    
    return best_val_loss

def evaluate_model():
    """Quick evaluation for demo"""
    try:
        checkpoint = torch.load("checkpoints/basket_transformer.pt", map_location='cpu')
        model = GPTModel(checkpoint['config']).to(CFG['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"\n🔮 MODEL EVALUATION:")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Train Loss: {checkpoint['train_loss']:.4f}")
        print(f"   Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"   Train Acc: {checkpoint['train_acc']:.4f}")
        print(f"   Val Acc: {checkpoint['val_acc']:.4f}")
        
        # Demo inference
        dummy_input = torch.randint(1, CFG['vocab_size'], (1, CFG['context_length']-1)).to(CFG['device'])
        
        with torch.no_grad():
            logits = model(dummy_input)[:, -1, :]
            top_k = torch.topk(logits, k=12, dim=-1)
            
        print(f"\n📈 Inference Demo:")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Top 12 predictions: {top_k.indices[0].cpu().tolist()}")
        print("   ✅ Model ready for recommendations!")
        
    except FileNotFoundError:
        print("❌ No saved model found")
    except Exception as e:
        print(f"❌ Evaluation error: {e}")

if __name__ == "__main__":
    torch.manual_seed(CFG['seed'])
    random.seed(CFG['seed'])
    
    # Training
    best_loss = train()
    
    # Evaluation
    evaluate_model()
    
    print(f"\n🚀 READY FOR DEMO!")
    print("=" * 30)
    print("Key Features:")
    print("• Transformer architecture for sequential recommendations")
    print("• 50-token context, 12K vocabulary")
    print("• Trained for next-purchase prediction")
    print("• Ready for MAP@12 evaluation")
    print(f"• Model size: ~{sum(p.numel() for p in GPTModel(CFG).parameters())/1e6:.1f}M parameters")
