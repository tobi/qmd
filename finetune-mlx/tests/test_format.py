#!/usr/bin/env python3
"""Tests for QMD output format validation."""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval import score_expansion


class TestOutputFormat:
    """Test output format scoring."""
    
    def test_perfect_format(self):
        """Test scoring of perfect format output."""
        text = """hyde: The Renaissance was a cultural movement in Europe.
lex: Renaissance period 14th century
lex: Renaissance art culture
vec: what was the Renaissance period
vec: how did the Renaissance transform art"""
        
        scores = score_expansion(text)
        assert scores["has_lex"] == 1
        assert scores["has_vec"] == 1
        assert scores["has_hyde"] == 1
        assert scores["format_valid"] == 1
        assert scores["total"] >= 100
    
    def test_missing_hyde(self):
        """Test output missing hyde component."""
        text = """lex: auth config settings
lex: authentication setup
vec: how to configure auth
vec: auth tutorial"""
        
        scores = score_expansion(text)
        assert scores["has_lex"] == 1
        assert scores["has_vec"] == 1
        assert scores["has_hyde"] == 0
        assert scores["format_valid"] == 0
    
    def test_missing_lex(self):
        """Test output missing lex component."""
        text = """hyde: To configure authentication, use the config file.
vec: how to set up auth
vec: auth configuration guide"""
        
        scores = score_expansion(text)
        assert scores["has_lex"] == 0
        assert scores["has_vec"] == 1
        assert scores["has_hyde"] == 1
        assert scores["format_valid"] == 0
    
    def test_missing_vec(self):
        """Test output missing vec component."""
        text = """hyde: Database connections require proper configuration.
lex: database connection
lex: db config"""
        
        scores = score_expansion(text)
        assert scores["has_lex"] == 1
        assert scores["has_vec"] == 0
        assert scores["has_hyde"] == 1
        assert scores["format_valid"] == 0
    
    def test_empty_output(self):
        """Test empty output."""
        scores = score_expansion("")
        assert scores["format_valid"] == 0
        assert scores["total"] == 0
    
    def test_multiple_lex_entries(self):
        """Test counting multiple lex entries."""
        text = """hyde: Test
lex: term1
lex: term2
lex: term3
lex: term4
vec: query"""
        
        scores = score_expansion(text)
        assert scores["lex_count"] == 4
        # Max 3 count toward score
        assert scores["total"] == 40 + 30 + 10 + 20  # format + 3*lex + 1*vec + hyde
    
    def test_multiple_vec_entries(self):
        """Test counting multiple vec entries."""
        text = """hyde: Test
lex: term
vec: query1
vec: query2
vec: query3
vec: query4"""
        
        scores = score_expansion(text)
        assert scores["vec_count"] == 4
        # Max 3 count toward score
        assert scores["total"] == 40 + 10 + 30 + 20  # format + 1*lex + 3*vec + hyde


class TestEdgeCases:
    """Test edge cases."""
    
    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        text = """  hyde: Some text  
  lex: keyword  
  vec: natural question  """
        
        scores = score_expansion(text)
        assert scores["has_lex"] == 1
        assert scores["has_vec"] == 1
        assert scores["has_hyde"] == 1
    
    def test_case_sensitivity(self):
        """Test that prefixes are case sensitive (lowercase required)."""
        text = """HYDE: Some text
LEX: keyword
VEC: question"""
        
        scores = score_expansion(text)
        # Uppercase should not match
        assert scores["has_lex"] == 0
        assert scores["has_vec"] == 0
        assert scores["has_hyde"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
