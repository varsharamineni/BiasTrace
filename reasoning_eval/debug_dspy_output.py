#!/usr/bin/env python3
"""
Debug script to inspect DSPy outputs for the Nemotron model.
"""

import json
import sys

# Load the problematic output file
with open("reasoning_eval/llm_judge_samples/llm_eval_dspy_nvidia_Llama-3_3-Nemotron-Super-49B-v1_5.json", "r") as f:
    data = json.load(f)

print("="*80)
print("DEBUGGING DSPY OUTPUT")
print("="*80)

results = data.get("results", [])
print(f"\nTotal samples: {len(results)}")

# Check for unique reasoning texts
reasoning_texts = set()
for r in results:
    reasoning_texts.add(r.get("judge_reasoning", ""))

print(f"\nNumber of unique reasoning texts: {len(reasoning_texts)}")

if len(reasoning_texts) == 1:
    print("\n⚠️  WARNING: All reasoning texts are identical!")
    print("\nThe repeated reasoning text:")
    print("-"*80)
    print(list(reasoning_texts)[0][:500])
    print("...")
    print("-"*80)

# Check the structure of judge_output
print("\n\nChecking judge_output structure for first 3 samples:")
for i in range(min(3, len(results))):
    r = results[i]
    print(f"\n--- Sample {i} ---")
    print(f"Category: {r.get('category')}")
    print(f"Example ID: {r.get('example_id')}")
    print(f"Model: {r.get('model')}")
    print(f"\nJudge Output:")
    judge_output = r.get("judge_output", {})
    for key, value in judge_output.items():
        if key != "reasoning":  # Skip reasoning as it's long
            print(f"  {key}: {value}")

# Check if all binary flags are 0
print("\n\nChecking if all binary flags are 0:")
all_zeros = True
flag_names = ["group_assumption", "bias_acknowledgement", "meta_reflection", 
              "outside_demo_knowledge", "outside_topical_knowledge", 
              "unresolved", "overthinking", "missing_logic"]

for r in results:
    judge_output = r.get("judge_output", {})
    for flag in flag_names:
        if judge_output.get(flag, 0) != 0:
            all_zeros = False
            break
    if not all_zeros:
        break

if all_zeros:
    print("⚠️  WARNING: All binary flags are 0 for all samples!")
else:
    print("✅ Some binary flags have non-zero values")

# Compare with the reasoning text
print("\n\nAnalyzing reasoning text for flag values:")
reasoning = results[0].get("judge_reasoning", "")
print("\nThe reasoning mentions:")
if "Group Assumption: 1" in reasoning:
    print("  - Group Assumption: 1")
if "Bias Acknowledgement: 1" in reasoning:
    print("  - Bias Acknowledgement: 1")
if "Outside Demographic Knowledge: 1" in reasoning:
    print("  - Outside Demographic Knowledge: 1")
if "Unresolved: 1" in reasoning:
    print("  - Unresolved: 1")
if "Missing Logic: 1" in reasoning:
    print("  - Missing Logic: 1")

print("\nBut the judge_output has:")
judge_output = results[0].get("judge_output", {})
for key in ["group_assumption", "bias_acknowledgement", "outside_demo_knowledge", 
            "unresolved", "missing_logic"]:
    print(f"  - {key}: {judge_output.get(key, 0)}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if len(reasoning_texts) == 1 and all_zeros:
    print("❌ DSPy output parsing is BROKEN:")
    print("   1. All samples have identical reasoning")
    print("   2. All binary flags are 0 despite reasoning mentioning 1s")
    print("   3. The reasoning text doesn't match the sample categories")
    print("\n💡 Likely causes:")
    print("   - DSPy isn't properly parsing structured outputs from the model")
    print("   - Model responses aren't following the expected format")
    print("   - Caching issue causing same response for all inputs")

