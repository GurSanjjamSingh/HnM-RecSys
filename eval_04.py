# Basket Recommender Demo & Evaluation - MAP@12
import os, json, random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_03 import GPTModel

class BasketEvaluator:
    def __init__(self, model_path="checkpoints/basket_transformer.pt"):
        """Load trained model and setup for evaluation"""
        print("🔮 Loading Basket Recommendation Model...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        self.config = checkpoint['config']
        self.vocab_map = checkpoint.get('vocab_map', None)
        
        # Load product mappings
        print("📋 Loading product mappings...")
        with open('data/mappings/idx2product.json', 'r') as f:
            self.id2product = json.load(f)
        with open('data/mappings/product2idx.json', 'r') as f:
            self.product2id = json.load(f)
        
        # Convert string keys to int for id2product
        self.id2product = {int(k): v for k, v in self.id2product.items()}
        
        # Initialize model
        self.model = GPTModel(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config['device'])
        self.model.eval()
        
        # Load test data for evaluation
        self.test_sequences = torch.load('data/splits/test_padded.pt', map_location='cpu')
        
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"📦 Product catalog: {len(self.id2product)} items")
        print(f"🧪 Test sequences: {len(self.test_sequences)} baskets")
    
    def original_to_model_id(self, original_id):
        """Convert original product ID to model vocabulary ID"""
        if self.vocab_map is None:
            return original_id
        return self.vocab_map.get(original_id, 0)  # 0 for unknown items
    
    def model_to_original_id(self, model_id):
        """Convert model vocabulary ID back to original product ID"""
        if self.vocab_map is None:
            return model_id
        
        # Reverse lookup in vocab_map
        for orig_id, mapped_id in self.vocab_map.items():
            if mapped_id == model_id:
                return orig_id
        return 0  # Unknown
    
    def get_product_name(self, product_id):
        """Get formatted product name"""
        if product_id == 0:
            return "🚫 [PADDING/UNKNOWN]"
        
        original_id = self.model_to_original_id(product_id) if self.vocab_map else product_id
        product_name = self.id2product.get(original_id, f"Product_{original_id}")
        
        # Add emoji based on product category/name
        name_lower = product_name.lower()
        if any(word in name_lower for word in ['milk', 'dairy', 'cheese', 'yogurt']):
            emoji = "🥛"
        elif any(word in name_lower for word in ['bread', 'bakery', 'roll', 'bagel']):
            emoji = "🍞"
        elif any(word in name_lower for word in ['apple', 'banana', 'fruit', 'orange', 'berry']):
            emoji = "🍎"
        elif any(word in name_lower for word in ['vegetable', 'carrot', 'lettuce', 'spinach']):
            emoji = "🥬"
        elif any(word in name_lower for word in ['meat', 'chicken', 'beef', 'pork']):
            emoji = "🍗"
        elif any(word in name_lower for word in ['beverage', 'drink', 'soda', 'juice']):
            emoji = "🥤"
        else:
            emoji = "🛒"
        
        return f"{emoji} {product_name}"
    
    def predict_next_items(self, basket_sequence, k=12):
        """Predict next k items for a basket sequence"""
        with torch.no_grad():
            # Prepare input (remove last item for prediction)
            input_seq = basket_sequence[:-1]
            
            # Pad/truncate to model's context length - 1
            max_len = self.config['context_length'] - 1
            if len(input_seq) > max_len:
                input_seq = input_seq[-max_len:]  # Keep recent items
            else:
                # Pad with zeros at beginning
                pad_len = max_len - len(input_seq)
                input_seq = torch.cat([torch.zeros(pad_len, dtype=torch.long), input_seq])
            
            # Add batch dimension and move to device
            input_tensor = input_seq.unsqueeze(0).to(self.config['device'])
            
            # Get predictions
            logits = self.model(input_tensor)[:, -1, :]  # Last position
            
            # Get top-k predictions
            top_k = torch.topk(logits, k=k, dim=-1)
            predicted_ids = top_k.indices[0].cpu().tolist()
            predicted_scores = top_k.values[0].cpu().tolist()
            
            return predicted_ids, predicted_scores
    
    def calculate_map_at_k(self, predictions, actual_items, k=12):
        """Calculate MAP@k for a single prediction"""
        if not actual_items:
            return 0.0
        
        # Take top k predictions
        pred_k = predictions[:k]
        
        # Calculate average precision
        relevant_items = set(actual_items)
        hits = 0
        precision_sum = 0.0
        
        for i, pred in enumerate(pred_k):
            if pred in relevant_items:
                hits += 1
                precision_sum += hits / (i + 1)
        
        if hits == 0:
            return 0.0
        
        return precision_sum / min(len(relevant_items), k)
    

    
    def comprehensive_evaluation(self, num_samples=100):
        """Comprehensive MAP@12 evaluation on test set"""
        print(f"\n🧪 COMPREHENSIVE EVALUATION (MAP@12 on {num_samples} samples)")
        print("="*60)
        
        # Sample test sequences
        if num_samples >= len(self.test_sequences):
            test_indices = list(range(len(self.test_sequences)))
        else:
            test_indices = random.sample(range(len(self.test_sequences)), num_samples)
        
        map_scores = []
        
        for idx in tqdm(test_indices, desc="Evaluating"):
            # Process sequence same as demonstration
            original_seq = self.test_sequences[idx]
            
            if self.vocab_map:
                mapped_seq = torch.tensor([self.vocab_map.get(item.item(), 0) for item in original_seq])
            else:
                mapped_seq = original_seq
            
            nonzero_mask = mapped_seq != 0
            if nonzero_mask.sum() < 2:
                continue
            
            nonzero_indices = nonzero_mask.nonzero().flatten()
            start_idx = nonzero_indices[0].item()
            actual_sequence = mapped_seq[start_idx:]
            
            if len(actual_sequence) < 2:
                continue
            
            target_items = [actual_sequence[-1].item()]
            predicted_ids, _ = self.predict_next_items(actual_sequence)
            
            map_score = self.calculate_map_at_k(predicted_ids, target_items, k=12)
            map_scores.append(map_score)
        
        # Results
        if map_scores:
            avg_map = np.mean(map_scores)
            std_map = np.std(map_scores)
            median_map = np.median(map_scores)
            
            print(f"\n📊 EVALUATION RESULTS:")
            print(f"   Samples Evaluated: {len(map_scores)}")
            print(f"   Average MAP@12: {avg_map:.4f} ± {std_map:.4f}")
            print(f"   Median MAP@12: {median_map:.4f}")
            print(f"   Min MAP@12: {min(map_scores):.4f}")
            print(f"   Max MAP@12: {max(map_scores):.4f}")
            
            # Distribution
            perfect_predictions = sum(1 for score in map_scores if score == 1.0)
            zero_predictions = sum(1 for score in map_scores if score == 0.0)
            
            print(f"\n📈 SCORE DISTRIBUTION:")
            print(f"   Perfect Predictions (MAP=1.0): {perfect_predictions} ({perfect_predictions/len(map_scores)*100:.1f}%)")
            print(f"   Zero Predictions (MAP=0.0): {zero_predictions} ({zero_predictions/len(map_scores)*100:.1f}%)")
            
            return avg_map, map_scores
        else:
            print("❌ No valid predictions found!")
            return 0.0, []
    
    def show_random_products(self, n=20):
        """Show random products from catalog"""
        print(f"\n📋 RANDOM PRODUCT SAMPLE ({n} items)")
        print("="*60)
        
        # Get random product IDs from the catalog
        available_ids = list(self.id2product.keys())
        if len(available_ids) < n:
            n = len(available_ids)
        
        random_ids = random.sample(available_ids, n)
        
        for i, prod_id in enumerate(random_ids):
            # Convert to model ID if vocabulary mapping exists
            model_id = self.original_to_model_id(prod_id)
            product_name = self.get_product_name(model_id if self.vocab_map else prod_id)
            
            print(f"{i+1:2d}. ID: {prod_id:4d} → {product_name}")

def main():
    """Main evaluation function"""
    print("🛒 BASKET RECOMMENDATION SYSTEM")
    print("🎯 MAP@12 Evaluation on Test Set")
    print("="*60)
    
    # Set random seed for reproducible results
    random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Initialize evaluator
        evaluator = BasketEvaluator()
        
        # Show random products
        evaluator.show_random_products(20)
        
        # Comprehensive evaluation
        avg_map, all_scores = evaluator.comprehensive_evaluation(num_samples=500)
        
        print(f"\n🎉 EVALUATION COMPLETE!")
        print("="*40)
        print(f"📈 Final MAP@12: {avg_map:.4f}")
        print(f"✅ Ready for production!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("📁 Please ensure the following files exist:")
        print("   - checkpoints/basket_transformer.pt")
        print("   - data/splits/test_padded.pt")
        print("   - data/mappings/id2product.json")
        print("   - data/mappings/product2id.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()