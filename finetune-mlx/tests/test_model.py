#!/usr/bin/env python3
"""Tests for model loading and inference."""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# These tests require the model to be downloaded
# Mark as slow/integration tests

@pytest.mark.slow
class TestModelLoading:
    """Test model loading functionality."""
    
    def test_base_model_exists(self):
        """Check if base model directory exists."""
        model_path = Path(__file__).parent.parent / "models" / "Qwen_Qwen2.5-1.5B" / "mlx"
        assert model_path.exists(), f"Base model not found at {model_path}"
    
    def test_adapter_exists(self):
        """Check if trained adapter exists."""
        adapter_path = Path(__file__).parent.parent / "adapters" / "sft-full"
        assert adapter_path.exists(), f"Adapter not found at {adapter_path}"
        
        # Check required files
        assert (adapter_path / "adapters.safetensors").exists()
        assert (adapter_path / "adapter_config.json").exists()
    
    def test_adapter_config_valid(self):
        """Validate adapter config JSON."""
        import json
        
        config_path = Path(__file__).parent.parent / "adapters" / "sft-full" / "adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Check required fields
        assert "lora_layers" in config
        assert "rank" in config or "r" in config


@pytest.mark.slow
class TestInference:
    """Test model inference."""
    
    @pytest.fixture
    def model_and_tokenizer(self):
        """Load model with adapter."""
        from mlx_lm import load
        
        model_path = Path(__file__).parent.parent / "models" / "Qwen_Qwen2.5-1.5B" / "mlx"
        adapter_path = Path(__file__).parent.parent / "adapters" / "sft-full"
        
        model, tokenizer = load(str(model_path), adapter_path=str(adapter_path))
        return model, tokenizer
    
    def test_generate_expansion(self, model_and_tokenizer):
        """Test generating a query expansion."""
        from mlx_lm import generate
        
        model, tokenizer = model_and_tokenizer
        
        prompt = """<|im_start|>user
/no_think Expand this search query: test query<|im_end|>
<|im_start|>assistant
"""
        
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=128,
            verbose=False,
        )
        
        assert len(response) > 0
        # Should contain at least some expected prefixes
        response_lower = response.lower()
        has_valid_output = (
            "lex:" in response_lower or 
            "vec:" in response_lower or 
            "hyde:" in response_lower
        )
        assert has_valid_output, f"Response missing expected format: {response}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
