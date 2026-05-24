"""
Simplified Transformer Setup Validation
Tests tokenizer loading for IndoBERTweet without requiring torch or datasets
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_tokenizer_only():
    """Test IndoBERTweet tokenizer loading (core requirement)"""
    print("=" * 70)
    print("PHASE 2A: TRANSFORMER TOKENIZER VALIDATION")
    print("=" * 70)
    
    try:
        print("\n[1] Testing transformers import...")
        from transformers import AutoTokenizer
        print("[OK] transformers.AutoTokenizer imported")
        
        print("\n[2] Loading IndoBERTweet tokenizer...")
        model_name = "indolem/indobertweet-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[OK] Tokenizer loaded: {type(tokenizer).__name__}")
        
        print("\n[3] Checking tokenizer properties...")
        print(f"  - Vocab size: {tokenizer.vocab_size:,}")
        print(f"  - Max length: {tokenizer.model_max_length}")
        print(f"  - Type: {type(tokenizer).__name__}")
        
        print("\n[4] Testing tokenization...")
        test_texts = [
            "Aplikasi ini sangat membantu",
            "Saya tidak suka aplikasi ini",
            "Bagus tapi perlu diperbaiki"
        ]
        
        for i, text in enumerate(test_texts, 1):
            tokens = tokenizer.tokenize(text)
            print(f"  Sample {i}: {len(tokens)} tokens from '{text}'")
        
        print("\n[5] Testing tokenizer with tensors...")
        from_pretrained_kwargs = {
            'return_tensors': 'pt',
            'padding': True,
            'truncation': True,
            'max_length': 512
        }
        
        try:
            encoded = tokenizer(test_texts[0], **from_pretrained_kwargs)
            print(f"[OK] Encoding successful")
            print(f"  - Input IDs shape: {encoded['input_ids'].shape}")
            print(f"  - Attention mask shape: {encoded['attention_mask'].shape}")
        except Exception as e:
            print(f"[WARNING] Tensor encoding requires PyTorch: {type(e).__name__}")
            print("  (This is expected if PyTorch is not installed)")
        
        print("\n" + "=" * 70)
        print("TOKENIZER VALIDATION: [PASS]")
        print("=" * 70)
        print("\nIndoBERTweet tokenizer is ready for use!")
        print("PyTorch installation is needed for model loading and inference.")
        return True
        
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        print("=" * 70)
        print("TOKENIZER VALIDATION: [FAIL]")
        print("=" * 70)
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        print("TOKENIZER VALIDATION: [FAIL]")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = test_tokenizer_only()
    sys.exit(0 if success else 1)
