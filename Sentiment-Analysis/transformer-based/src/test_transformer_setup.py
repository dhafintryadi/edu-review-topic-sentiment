"""
Transformer Setup Validation Script
Test script to validate HuggingFace transformer integration and IndoBERTweet model loading
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LOGS_DIR


class TransformerSetupValidator:
    """Validate transformer ecosystem setup"""
    
    def __init__(self):
        """Initialize validator"""
        self.results = {}
        self.model_name = "indolem/indobertweet-base-uncased"
        
    def test_imports(self) -> bool:
        """Test basic transformer imports"""
        logger.info("=" * 70)
        logger.info("STEP 1: Testing Transformer Imports")
        logger.info("=" * 70)
        
        try:
            logger.info("Importing transformers...")
            from transformers import AutoTokenizer, AutoModel, AutoConfig
            logger.info("[OK] transformers imported successfully")
            
            logger.info("Importing datasets...")
            import datasets
            logger.info("[OK] datasets imported successfully")
            
            logger.info("Importing accelerate...")
            import accelerate
            logger.info("[OK] accelerate imported successfully")
            
            logger.info("Importing torch...")
            import torch
            logger.info(f"[OK] torch imported successfully (version: {torch.__version__})")
            
            logger.info("Importing evaluate...")
            import evaluate
            logger.info("[OK] evaluate imported successfully")
            
            self.results['imports'] = {'success': True}
            logger.info("[OK] ALL IMPORTS SUCCESSFUL\n")
            return True
            
        except ImportError as e:
            logger.error(f"[ERROR] Import error: {e}")
            self.results['imports'] = {'success': False, 'error': str(e)}
            return False
    
    def test_tokenizer_loading(self) -> bool:
        """Test IndoBERTweet tokenizer loading"""
        logger.info("=" * 70)
        logger.info("STEP 2: Testing IndoBERTweet Tokenizer Loading")
        logger.info("=" * 70)
        
        try:
            from transformers import AutoTokenizer
            
            logger.info(f"Loading tokenizer: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("[OK] Tokenizer loaded successfully")
            
            # Get tokenizer info
            vocab_size = tokenizer.vocab_size
            logger.info(f"[OK] Vocab size: {vocab_size}")
            
            model_max_length = tokenizer.model_max_length
            logger.info(f"[OK] Max length: {model_max_length}")
            
            logger.info(f"[OK] Tokenizer type: {type(tokenizer).__name__}")
            
            self.results['tokenizer'] = {
                'success': True,
                'vocab_size': vocab_size,
                'model_max_length': model_max_length,
                'tokenizer_type': type(tokenizer).__name__
            }
            
            logger.info("[OK] TOKENIZER LOADING SUCCESSFUL\n")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Tokenizer loading error: {e}")
            self.results['tokenizer'] = {'success': False, 'error': str(e)}
            return False
    
    def test_model_loading(self) -> bool:
        """Test IndoBERTweet model loading"""
        logger.info("=" * 70)
        logger.info("STEP 3: Testing IndoBERTweet Model Loading")
        logger.info("=" * 70)
        
        try:
            from transformers import AutoModel, AutoConfig
            import torch
            
            logger.info(f"Loading model: {self.model_name}")
            model = AutoModel.from_pretrained(self.model_name)
            logger.info("[OK] Model loaded successfully")
            
            # Get model config
            config = AutoConfig.from_pretrained(self.model_name)
            logger.info(f"[OK] Config loaded successfully")
            
            # Get model info
            hidden_size = config.hidden_size
            num_attention_heads = config.num_attention_heads
            num_hidden_layers = config.num_hidden_layers
            vocab_size = config.vocab_size
            
            logger.info(f"[OK] Hidden size: {hidden_size}")
            logger.info(f"[OK] Number of attention heads: {num_attention_heads}")
            logger.info(f"[OK] Number of hidden layers: {num_hidden_layers}")
            logger.info(f"[OK] Vocab size: {vocab_size}")
            
            # Check device support
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[OK] Available device: {device}")
            if device == "cuda":
                logger.info(f"[OK] GPU available: {torch.cuda.get_device_name(0)}")
            
            model.to(device)
            logger.info(f"[OK] Model moved to {device}")
            
            # Get model size
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"[OK] Total parameters: {param_count:,}")
            
            self.results['model'] = {
                'success': True,
                'hidden_size': hidden_size,
                'num_attention_heads': num_attention_heads,
                'num_hidden_layers': num_hidden_layers,
                'vocab_size': vocab_size,
                'device': device,
                'total_parameters': param_count,
                'model_type': config.model_type
            }
            
            logger.info("[OK] MODEL LOADING SUCCESSFUL\n")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Model loading error: {e}")
            self.results['model'] = {'success': False, 'error': str(e)}
            return False
    
    def test_tokenization(self) -> bool:
        """Test tokenization process"""
        logger.info("=" * 70)
        logger.info("STEP 4: Testing Tokenization")
        logger.info("=" * 70)
        
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Test sentences
            test_sentences = [
                "Aplikasi ini sangat membantu untuk belajar",
                "Saya tidak suka aplikasi ini",
                "Aplikasi bagus tapi banyak yang perlu diperbaiki"
            ]
            
            logger.info("Testing tokenization with sample sentences:")
            for i, sentence in enumerate(test_sentences, 1):
                logger.info(f"\n  Sample {i}: {sentence}")
                
                tokens = tokenizer.tokenize(sentence)
                logger.info(f"    Tokens: {tokens}")
                
                encoding = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
                logger.info(f"    Input IDs shape: {encoding['input_ids'].shape}")
                logger.info(f"    Token count: {len(encoding['input_ids'][0])}")
            
            self.results['tokenization'] = {'success': True}
            logger.info("\n[OK] TOKENIZATION TEST SUCCESSFUL\n")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Tokenization error: {e}")
            self.results['tokenization'] = {'success': False, 'error': str(e)}
            return False
    
    def test_inference(self) -> bool:
        """Test model inference"""
        logger.info("=" * 70)
        logger.info("STEP 5: Testing Model Inference")
        logger.info("=" * 70)
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            # Test inference
            test_text = "Aplikasi ini sangat membantu untuk belajar"
            logger.info(f"Test text: {test_text}")
            
            with torch.no_grad():
                inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                logger.info(f"[OK] Input prepared on {device}")
                logger.info(f"[OK] Input keys: {list(inputs.keys())}")
                
                outputs = model(**inputs)
                
                logger.info(f"[OK] Model inference successful")
                logger.info(f"[OK] Last hidden state shape: {outputs.last_hidden_state.shape}")
                logger.info(f"[OK] Pooler output shape: {outputs.pooler_output.shape}")
            
            self.results['inference'] = {'success': True}
            logger.info("[OK] INFERENCE TEST SUCCESSFUL\n")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Inference error: {e}")
            self.results['inference'] = {'success': False, 'error': str(e)}
            return False
    
    def print_summary(self) -> None:
        """Print validation summary"""
        logger.info("=" * 70)
        logger.info("TRANSFORMER SETUP VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        print("\n" + "=" * 70)
        print("TRANSFORMER SETUP VALIDATION SUMMARY")
        print("=" * 70)
        
        # Imports
        imports_status = "[PASS]" if self.results.get('imports', {}).get('success', False) else "[FAIL]"
        print(f"\n1. Imports: {imports_status}")
        
        # Tokenizer
        tokenizer_status = "[PASS]" if self.results.get('tokenizer', {}).get('success') else "[FAIL]"
        print(f"\n2. Tokenizer Loading: {tokenizer_status}")
        if self.results.get('tokenizer', {}).get('success'):
            print(f"   - Vocab size: {self.results['tokenizer']['vocab_size']:,}")
            print(f"   - Max length: {self.results['tokenizer']['model_max_length']}")
            print(f"   - Type: {self.results['tokenizer']['tokenizer_type']}")
        
        # Model
        model_status = "[PASS]" if self.results.get('model', {}).get('success') else "[FAIL]"
        print(f"\n3. Model Loading: {model_status}")
        if self.results.get('model', {}).get('success'):
            print(f"   - Hidden size: {self.results['model']['hidden_size']}")
            print(f"   - Attention heads: {self.results['model']['num_attention_heads']}")
            print(f"   - Layers: {self.results['model']['num_hidden_layers']}")
            print(f"   - Vocab size: {self.results['model']['vocab_size']:,}")
            print(f"   - Device: {self.results['model']['device']}")
            print(f"   - Parameters: {self.results['model']['total_parameters']:,}")
            print(f"   - Type: {self.results['model']['model_type']}")
        
        # Tokenization
        tokenization_status = "[PASS]" if self.results.get('tokenization', {}).get('success') else "[FAIL]"
        print(f"\n4. Tokenization: {tokenization_status}")
        
        # Inference
        inference_status = "[PASS]" if self.results.get('inference', {}).get('success') else "[FAIL]"
        print(f"\n5. Model Inference: {inference_status}")
        
        # Overall
        all_passed = all(
            self.results.get(key, {}).get('success', False) if isinstance(self.results.get(key), dict) else False
            for key in ['imports', 'tokenizer', 'model', 'tokenization', 'inference']
        )
        
        overall_status = "[ALL TESTS PASSED]" if all_passed else "[SOME TESTS FAILED]"
        print(f"\nOVERALL STATUS: {overall_status}")
        print("\n" + "=" * 70 + "\n")
        
        logger.info(f"\nOVERALL STATUS: {overall_status}")
        logger.info("=" * 70 + "\n")
        
        return all_passed
    
    def run_all_tests(self) -> bool:
        """Run all validation tests"""
        logger.info("\n" + "=" * 70)
        logger.info("TRANSFORMER ECOSYSTEM SETUP VALIDATION")
        logger.info("=" * 70 + "\n")
        
        # Run all tests
        test1 = self.test_imports()
        test2 = self.test_tokenizer_loading() if test1 else False
        test3 = self.test_model_loading() if test2 else False
        test4 = self.test_tokenization() if test3 else False
        test5 = self.test_inference() if test4 else False
        
        # Print summary
        all_passed = self.print_summary()
        
        return all_passed


def main():
    """Main entry point"""
    try:
        validator = TransformerSetupValidator()
        success = validator.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
