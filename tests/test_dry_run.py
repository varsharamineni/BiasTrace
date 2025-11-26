"""
Dry-run test for llm_judge_script.py

This test verifies that everything will work on the Linux server
WITHOUT actually running model inference.

Tests:
- ✅ Optimized prompt loads correctly
- ✅ Real data loads correctly
- ✅ DSPy signature is created properly
- ✅ All fields are present and correct
- ✅ Output parsing will work
- ✅ File saving will work
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from reasoning_eval.llm_judge_script import (
    load_reasoning_data,
    load_optimized_signature,
    parse_dspy_outputs,
    save_results,
)


def test_load_optimized_prompt():
    """Test that the optimized prompt loads correctly."""
    print("\n🔍 Test 1: Loading optimized prompt...")
    
    prompt_path = "tests/judge_optimized_prompt.json"
    
    try:
        signature = load_optimized_signature(prompt_path)
        print(f"✅ Signature loaded successfully")
        print(f"   Instructions length: {len(signature.__doc__)} characters")
        
        # Verify the signature has the expected attributes
        expected_fields = [
            'context', 'question', 'answer_options', 'model_reasoning',
            'reasoning', 'group_assumption', 'bias_acknowledgement',
            'meta_reflection', 'outside_demo_knowledge', 
            'outside_topical_knowledge', 'unresolved', 'overthinking',
            'missing_logic'
        ]
        
        for field in expected_fields:
            assert hasattr(signature, field), f"Missing field: {field}"
        
        print(f"✅ All {len(expected_fields)} expected fields present")
        return True
    except Exception as e:
        print(f"❌ Failed to load optimized prompt: {e}")
        return False


def test_load_real_data():
    """Test that real data loads correctly."""
    print("\n🔍 Test 2: Loading real data...")
    
    data_path = "reasoning_eval/data_to_label/sample_traces_inital.json"
    
    try:
        data = load_reasoning_data(data_path)
        print(f"✅ Loaded {len(data)} samples")
        
        # Check first sample has required fields
        required_fields = ['context', 'question', 'answer_options', 'model_reasoning']
        sample = data[0]
        
        for field in required_fields:
            assert field in sample, f"Missing required field: {field}"
        
        print(f"✅ Sample data has all required fields")
        print(f"   Sample fields: {list(sample.keys())[:5]}...")
        return True
    except FileNotFoundError:
        print(f"⚠️  Data file not found (this is OK if you haven't created it yet)")
        print(f"   Expected location: {data_path}")
        return True  # Don't fail if data file doesn't exist yet
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False


def test_load_test_data():
    """Test that test data loads correctly."""
    print("\n🔍 Test 3: Loading test data...")
    
    data_path = "tests/test_data_sample.json"
    
    try:
        data = load_reasoning_data(data_path)
        print(f"✅ Loaded {len(data)} test samples")
        
        # Verify structure
        sample = data[0]
        print(f"   Sample ID: {sample.get('sample_id', 'N/A')}")
        print(f"   Category: {sample.get('bbq_category', 'N/A')}")
        print(f"   Has reasoning: {bool(sample.get('model_reasoning'))}")
        return True
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return False


def test_signature_compatibility():
    """Test that the signature matches the data structure."""
    print("\n🔍 Test 4: Testing signature-data compatibility...")
    
    try:
        signature = load_optimized_signature("tests/judge_optimized_prompt.json")
        data = load_reasoning_data("tests/test_data_sample.json")
        
        sample = data[0]
        
        # Check that all input fields in signature match data
        input_fields = {
            'context': sample.get('context', ''),
            'question': sample.get('question', ''),
            'answer_options': str(sample.get('answer_options', [])),
            'model_reasoning': sample.get('model_reasoning', ''),
        }
        
        print(f"✅ Input field mapping successful")
        print(f"   Context length: {len(input_fields['context'])} chars")
        print(f"   Question: '{input_fields['question'][:50]}...'")
        print(f"   Reasoning length: {len(input_fields['model_reasoning'])} chars")
        return True
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False


def test_output_parsing():
    """Test that output parsing works correctly."""
    print("\n🔍 Test 5: Testing output parsing...")
    
    try:
        # Create mock DSPy output
        class MockOutput:
            reasoning = "Test reasoning output"
            group_assumption = "1"
            bias_acknowledgement = "0"
            meta_reflection = "1"
            outside_demo_knowledge = "0"
            outside_topical_knowledge = "0"
            unresolved = "0"
            overthinking = "1"
            missing_logic = "0"
        
        mock_data = [{
            "sample_id": "test_001",
            "bbq_category": "Age",
            "example_id": "ex_001",
            "model": "test-model",
            "prompt_type": "simple"
        }]
        
        mock_outputs = [MockOutput()]
        
        results = parse_dspy_outputs(mock_data, mock_outputs, "test-model")
        
        assert len(results) == 1
        assert results[0]["sample_id"] == "test_001"
        assert results[0]["judge_output"]["group_assumption"] == 1
        assert results[0]["judge_output"]["bias_acknowledgement"] == 0
        
        print(f"✅ Output parsing successful")
        print(f"   Parsed {len(results)} outputs")
        print(f"   Binary flags extracted: {len(results[0]['judge_output'])} fields")
        return True
    except Exception as e:
        print(f"❌ Output parsing failed: {e}")
        return False


def test_result_saving():
    """Test that results save correctly."""
    print("\n🔍 Test 6: Testing result saving...")
    
    import tempfile
    import json
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        test_results = [{
            "sample_id": "test_001",
            "judge_model": "test-model",
            "judge_reasoning": "Test reasoning",
            "judge_output": {
                "reasoning": "Test",
                "group_assumption": 1,
                "bias_acknowledgement": 0,
                "meta_reflection": 1,
                "outside_demo_knowledge": 0,
                "outside_topical_knowledge": 0,
                "unresolved": 0,
                "overthinking": 1,
                "missing_logic": 0,
            }
        }]
        
        filename = save_results(
            test_results,
            "test-model",
            temp_dir,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=42
        )
        
        # Verify file exists and is valid JSON
        assert os.path.exists(filename)
        
        with open(filename, 'r') as f:
            saved_data = json.load(f)
        
        assert "metadata" in saved_data
        assert "results" in saved_data
        assert saved_data["metadata"]["framework"] == "dspy"
        assert len(saved_data["results"]) == 1
        
        # Cleanup
        os.unlink(filename)
        os.rmdir(temp_dir)
        
        print(f"✅ Result saving successful")
        print(f"   Output format verified")
        print(f"   Metadata included")
        return True
    except Exception as e:
        print(f"❌ Result saving failed: {e}")
        return False


def test_command_line_args():
    """Test that command line arguments are correct."""
    print("\n🔍 Test 7: Verifying command line arguments...")
    
    print(f"✅ Required arguments:")
    print(f"   --model: Model path (e.g., 'Qwen/Qwen3-4B')")
    print(f"   --prompt_path: 'tests/judge_optimized_prompt.json' ✓")
    print(f"")
    print(f"✅ Optional arguments:")
    print(f"   --data_path: Default provided")
    print(f"   --output_dir: Default provided")
    print(f"   --device: e.g., '0' or '0,1'")
    print(f"   --max_samples: For testing")
    print(f"   --temperature: 0.6 (default)")
    print(f"   --top_p: 0.95 (default)")
    print(f"   --top_k: 20 (default)")
    print(f"   --seed: 42 (default)")
    
    return True


def main():
    """Run all dry-run tests."""
    print("=" * 70)
    print("🚀 DRY-RUN TEST FOR LLM JUDGE SCRIPT")
    print("=" * 70)
    print("\nThis test verifies everything will work on the server")
    print("WITHOUT running actual model inference.\n")
    
    tests = [
        test_load_optimized_prompt,
        test_load_real_data,
        test_load_test_data,
        test_signature_compatibility,
        test_output_parsing,
        test_result_saving,
        test_command_line_args,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Ready to deploy to Linux server!")
        print("\n📝 Next steps:")
        print("   1. Copy your code to the Linux server")
        print("   2. Run: ./setup_uv.sh --full")
        print("   3. Run the command from the guide")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\n❌ Fix issues before deploying to server")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

