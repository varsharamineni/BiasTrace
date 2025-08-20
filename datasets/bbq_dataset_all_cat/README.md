---
license: cc-by-4.0
---
# BBQ
Repository for the Bias Benchmark for QA dataset.
https://github.com/nyu-mll/BBQ 

Authors: Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel R. Bowman.

## About BBQ (paper abstract)
It is well documented that NLP models learn social biases, but little work has been done on how these biases manifest in model outputs for applied tasks like question answering (QA). We introduce the Bias Benchmark for QA (BBQ), a dataset of question sets constructed by the authors that highlight attested social biases against people belonging to protected classes along nine social dimensions relevant for U.S. English-speaking contexts. Our task evaluates model responses at two levels: (i) given an under-informative context, we test how strongly responses refect social biases, and (ii) given an adequately informative context, we test whether the model's biases override a correct answer choice. We fnd that models often rely on stereotypes when the context is under-informative, meaning the model's outputs consistently reproduce harmful biases in this setting. Though models are more accurate when the context provides an informative answer, they still rely on stereotypes and average up to 3.4 percentage points higher accuracy when the correct answer aligns with a social bias than when it conficts, with this difference widening to over 5 points on examples targeting gender for most models tested.

## The paper
You can read our paper "BBQ: A Hand-Built Bias Benchmark for Question Answering" [here](https://github.com/nyu-mll/BBQ/blob/main/QA_bias_benchmark.pdf). The paper has been published in the Findings of ACL 2022 [here](https://aclanthology.org/2022.findings-acl.165/).
