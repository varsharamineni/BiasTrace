"""
Test suite for llm_judge_script.py

This module tests the LLM judge script functionality without requiring
actual model inference by mocking DSPy components.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the functions we want to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reasoning_eval.llm_judge_script import (
    load_reasoning_data,
    load_optimized_signature,
    parse_dspy_outputs,
    save_results,
)


class TestLoadReasoningData(unittest.TestCase):
    """Test loading reasoning data from JSON."""

    def setUp(self):
        """Create temporary test data file."""
        self.test_data = [
            {
                "sample_id": "test_001",
                "model_reasoning": "This is test reasoning",
                "model_answer": "Option A",
                "bbq_category": "Age",
                "context": "Test context",
                "question": "Test question?",
                "answer_options": ["Option A", "Option B", "Option C"],
                "example_id": "ex_001",
                "model": "test-model",
                "prompt_type": "simple"
            }
        ]
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.test_data, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)

    def test_load_reasoning_data(self):
        """Test that reasoning data loads correctly."""
        data = load_reasoning_data(self.temp_file.name)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["sample_id"], "test_001")
        self.assertEqual(data[0]["model_reasoning"], "This is test reasoning")


class TestLoadOptimizedSignature(unittest.TestCase):
    """Test loading DSPy optimized signature."""

    def setUp(self):
        """Create temporary optimized prompt file."""
        self.optimized_prompt = {
            "predict": {
                "traces": [],
                "train": [],
                "demos": [],
                "signature": {
                    "instructions": "Test instructions for evaluation",
                    "fields": [
                        {"prefix": "Context:", "description": "${context}"},
                        {"prefix": "Question:", "description": "${question}"},
                        {"prefix": "Answer Options:", "description": "${answer_options}"},
                        {"prefix": "Model Reasoning:", "description": "${model_reasoning}"},
                        {"prefix": "Reasoning:", "description": "${reasoning}"},
                        {"prefix": "Group Assumption:", "description": "Binary (0 or 1)"},
                        {"prefix": "Bias Acknowledgement:", "description": "Binary (0 or 1)"},
                        {"prefix": "Meta Reflection:", "description": "Binary (0 or 1)"},
                        {"prefix": "Outside Demo Knowledge:", "description": "Binary (0 or 1)"},
                        {"prefix": "Outside Topical Knowledge:", "description": "Binary (0 or 1)"},
                        {"prefix": "Unresolved:", "description": "Binary (0 or 1)"},
                        {"prefix": "Overthinking:", "description": "Binary (0 or 1)"},
                        {"prefix": "Missing Logic:", "description": "Binary (0 or 1)"}
                    ]
                },
                "lm": None
            }
        }
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
        json.dump(self.optimized_prompt, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)

    @patch('reasoning_eval.llm_judge_script.dspy')
    def test_load_optimized_signature(self, mock_dspy):
        """Test that optimized signature loads correctly."""
        # Mock dspy.Signature
        mock_signature_class = MagicMock()
        mock_dspy.Signature = mock_signature_class
        mock_dspy.InputField = MagicMock(side_effect=lambda **kwargs: f"InputField({kwargs})")
        mock_dspy.OutputField = MagicMock(side_effect=lambda **kwargs: f"OutputField({kwargs})")

        signature = load_optimized_signature(self.temp_file.name)
        
        # Verify signature was created
        self.assertIsNotNone(signature)
        # Verify instructions were set
        self.assertEqual(signature.__doc__, "Test instructions for evaluation")


class TestParseDspyOutputs(unittest.TestCase):
    """Test parsing DSPy model outputs."""

    def setUp(self):
        """Create mock data and outputs."""
        self.batch_data = [
            {
                "sample_id": "test_001",
                "bbq_category": "Age",
                "example_id": "ex_001",
                "model": "test-model",
                "prompt_type": "simple"
            }
        ]
        
        # Mock DSPy output object
        self.mock_output = Mock()
        self.mock_output.reasoning = "This is the judge's reasoning"
        self.mock_output.group_assumption = "1"
        self.mock_output.bias_acknowledgement = "0"
        self.mock_output.meta_reflection = "1"
        self.mock_output.outside_demo_knowledge = "0"
        self.mock_output.outside_topical_knowledge = "0"
        self.mock_output.unresolved = "0"
        self.mock_output.overthinking = "1"
        self.mock_output.missing_logic = "0"

    def test_parse_dspy_outputs(self):
        """Test parsing DSPy outputs into structured format."""
        results = parse_dspy_outputs(
            self.batch_data,
            [self.mock_output],
            "test-judge-model"
        )
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        # Check metadata
        self.assertEqual(result["sample_id"], "test_001")
        self.assertEqual(result["judge_model"], "test-judge-model")
        
        # Check judge output
        self.assertEqual(result["judge_reasoning"], "This is the judge's reasoning")
        self.assertEqual(result["judge_output"]["group_assumption"], 1)
        self.assertEqual(result["judge_output"]["bias_acknowledgement"], 0)
        self.assertEqual(result["judge_output"]["meta_reflection"], 1)
        self.assertEqual(result["judge_output"]["overthinking"], 1)


class TestSaveResults(unittest.TestCase):
    """Test saving results to JSON."""

    def setUp(self):
        """Create temporary output directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_results = [
            {
                "sample_id": "test_001",
                "judge_model": "test-model",
                "judge_reasoning": "Test reasoning",
                "judge_output": {
                    "reasoning": "Test reasoning",
                    "group_assumption": 1,
                    "bias_acknowledgement": 0
                }
            }
        ]

    def test_save_results(self):
        """Test that results save correctly."""
        filename = save_results(
            self.test_results,
            "test-model",
            self.temp_dir,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=42
        )
        
        # Check file was created
        self.assertTrue(os.path.exists(filename))
        
        # Load and verify contents
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.assertIn("metadata", data)
        self.assertIn("results", data)
        self.assertEqual(data["metadata"]["judge_model"], "test-model")
        self.assertEqual(data["metadata"]["framework"], "dspy")
        self.assertEqual(len(data["results"]), 1)
        
        # Clean up
        os.unlink(filename)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    @patch('reasoning_eval.llm_judge_script.dspy')
    @patch('reasoning_eval.llm_judge_script.LM')
    def test_full_pipeline_mock(self, mock_lm, mock_dspy):
        """Test the full pipeline with mocked DSPy components."""
        # This is a placeholder for integration testing
        # In practice, you'd mock the entire chain and verify data flow
        
        # Mock the LM
        mock_lm_instance = MagicMock()
        mock_lm.return_value = mock_lm_instance
        
        # Mock dspy.configure
        mock_dspy.configure = MagicMock()
        
        # Mock ChainOfThought
        mock_cot = MagicMock()
        mock_dspy.ChainOfThought = MagicMock(return_value=mock_cot)
        
        # Verify mocks are set up correctly
        self.assertIsNotNone(mock_lm_instance)
        self.assertIsNotNone(mock_cot)


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestLoadReasoningData))
    suite.addTests(loader.loadTestsFromTestCase(TestLoadOptimizedSignature))
    suite.addTests(loader.loadTestsFromTestCase(TestParseDspyOutputs))
    suite.addTests(loader.loadTestsFromTestCase(TestSaveResults))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)

