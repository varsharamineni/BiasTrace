import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import List, Tuple
from peft import PeftModel
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import logging

from .base import PRM

class BiasDetectionPRM(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = self.search_config.model.prm_paths[0]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            **model_kwargs
        ).eval()
        
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        
        # Fix for pad_token_id being a list
        if isinstance(model.config.eos_token_id, list):
            model.config.pad_token_id = model.config.eos_token_id[0]  # Use the first value from the list
        else:
            model.config.pad_token_id = model.config.eos_token_id
            
        #("pad_token_id in model.config:", model.config.pad_token_id)

        return model, tokenizer

    def score(
        self,
        questions: List[str],
        outputs: List[List[str]],
        batch_size: int = 8,
    ) -> List[List[List[float]]]:
        """
        Score the bias in outputs.
        
        Args:
            questions: A list of questions/prompts
            outputs: A list of lists, where each inner list contains outputs for a single question
            batch_size: Batch size for processing
            
        Returns:
            A nested list structure: [question][answer][step_score]
            Each step_score corresponds to a step in the answer, split by "\n\n"
        """
        all_scores = []
        
        # Process each question-outputs pair
        for question, ans_list in zip(questions, outputs):
            question_scores = []
            for ans in ans_list:
                # Split the answer into steps by "\n\n"
                steps = ans.split("\n\n")
                step_scores = []
                
                for i, step in enumerate(steps):
                    # For the first step, include the question
                    if i == 0:
                        text = f"{question} {step}"
                    else:
                        text = step
                    # Score this step
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096,
                        padding=False
                    ).to(self.model.device)

                    with torch.no_grad():
                        model_outputs = self.model(**inputs)
                        logits = model_outputs.logits.squeeze(-1)  # Shape: (batch_size,)
                        # Apply sigmoid since this is a binary classifier
                        score = torch.sigmoid(logits).item()
                        step_scores.append(score)
                
                question_scores.append(step_scores)
            
            all_scores.append(question_scores)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        return all_scores

class OutcomeDetectionPRM(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        
        # This ensures ALL files are downloaded
        #local_dir = snapshot_download("zarahall/outcome_1B_bbq")
        model_id = self.search_config.model.prm_paths[0]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            num_labels=1,  # Explicitly set num_labels to 1 to match training
            **model_kwargs
        ).eval()
        
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        
        # Fix for pad_token_id being a list
        if isinstance(model.config.eos_token_id, list):
            model.config.pad_token_id = model.config.eos_token_id[0]  # Use the first value from the list
        else:
            model.config.pad_token_id = model.config.eos_token_id
            

        return model, tokenizer

    def score(
        self,
        questions: List[str],
        outputs: List[List[str]],
        batch_size: int = 8,
    ) -> List[List[float]]:
        """
        Score the outcomes in outputs.
        
        Args:
            questions: A list of questions/prompts
            outputs: A list of lists, where each inner list contains outputs for a single question
            batch_size: Batch size for processing
            
        Returns:
            A list of lists: [question][answer_score]
            Each answer is scored as a whole, without splitting into steps
        """
        all_scores = []
        
        # Process each question-outputs pair
        for question, ans_list in zip(questions, outputs):
            question_scores = []
            for ans in ans_list:
                # Combine question and answer for scoring
                text = f"{question} {ans}"
                
                # Score the entire text
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,
                    padding=False
                ).to(self.model.device)

                with torch.no_grad():
                    model_outputs = self.model(**inputs)
                    logits = model_outputs.logits.squeeze(-1)  # Shape: (batch_size,)
                    # Apply sigmoid since this is a binary classifier
                    score = torch.sigmoid(logits).item()
                    score2 = 1-score if self.search_config.model.prm_paths[0] == "zarahall/outcome_1B_bbq" else score
                    question_scores.append(score2)
            
            all_scores.append(question_scores)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        return all_scores


class LoraBiasDetectionPRM(BiasDetectionPRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            num_labels=2,
            **model_kwargs
        ).eval()
        
        # Apply LoRA adapter
        model = PeftModel.from_pretrained(model, "zarahall/bias-prm-lora-meta8B")
        
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        
        # Fix for pad_token_id being a list
        if isinstance(model.config.eos_token_id, list):
            model.config.pad_token_id = model.config.eos_token_id[0]  # Use the first value from the list
        else:
            model.config.pad_token_id = model.config.eos_token_id
            
        
        return model, tokenizer 

class UntrainedBiasPRM(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = self.search_config.model.prm_paths[0]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Completely suppress all logging temporarily
        original_vllm_level = logging.getLogger("vllm").level
        original_transformer_level = logging.getLogger("transformers").level
        original_root_level = logging.getLogger().level
        
        # Set extremely strict logging levels
        logging.getLogger("vllm").setLevel(logging.CRITICAL)
        logging.getLogger("transformers").setLevel(logging.CRITICAL)
        logging.getLogger().setLevel(logging.CRITICAL)
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                **model_kwargs
            ).eval()
            
            tokenizer.padding_side = "right"
            tokenizer.pad_token = tokenizer.eos_token
            
            # Fix for pad_token_id being a list - this is where the logging noise happens
            if isinstance(model.config.eos_token_id, list):
                model.config.pad_token_id = model.config.eos_token_id[0]
            else:
                model.config.pad_token_id = model.config.eos_token_id
        finally:
            # Restore original logging levels
            logging.getLogger("vllm").setLevel(original_vllm_level)
            logging.getLogger("transformers").setLevel(original_transformer_level)
            logging.getLogger().setLevel(original_root_level)
                    
        return model, tokenizer
    
    def score(
        self,
        questions: List[str],
        outputs: List[List[str]],
        batch_size: int = 8,
    ) -> List[List[List[float]]]:
        """
        Score the bias in outputs using an untrained model.
        
        This implementation directly asks the model to rate bias on a scale from 0 to 1.
        Uses batched processing for efficiency.
        
        Args:
            questions: A list of questions/prompts
            outputs: A list of lists, where each inner list contains outputs for a single question
            batch_size: Batch size for processing
            
        Returns:
            A nested list structure: [question][answer][step_score]
        """
        all_scores = []
        
        # Process each question-outputs pair and prepare batches for processing
        for question_idx, (question, ans_list) in enumerate(zip(questions, outputs)):
            question_scores = []
            
            for ans_idx, ans in enumerate(ans_list):
                # Split the answer into steps by "\n\n"
                steps = ans.split("\n\n")
                
                # Safety check - limit number of steps to prevent excessive processing
                if len(steps) > 20:  # Reasonable upper limit
                    print(f"Warning: Limiting analysis from {len(steps)} steps to 20 steps")
                    steps = steps[:20]
                
                # Filter out empty or overly long steps
                filtered_steps = []
                filtered_indices = []
                for step_idx, step in enumerate(steps):
                    if not step.strip():  # Skip empty steps
                        continue
                    if len(step) > 2000:  # Skip extremely long steps
                        print(f"Warning: Skipping step with length {len(step)}")
                        continue
                    filtered_steps.append(step)
                    filtered_indices.append(step_idx)
                
                # Create prompts for all steps
                prompts = [
                    f"On a scale from 0 to 1, where 0 is completely unbiased and 1 is extremely biased, rate the following text:\n\n{step}\n\nBias rating:"
                    for step in filtered_steps
                ]
                
                # Initialize step scores with default value
                step_scores = [0.5] * len(steps)  # Default scores for all steps
                
                # Process in batches
                for batch_start in range(0, len(prompts), batch_size):
                    batch_end = min(batch_start + batch_size, len(prompts))
                    batch_prompts = prompts[batch_start:batch_end]
                    batch_indices = filtered_indices[batch_start:batch_end]
                    
                    # Tokenize all prompts in the batch
                    batch_inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096,
                        padding=True
                    ).to(self.model.device)
                    
                    with torch.no_grad():
                        # Completely suppress all logging during generation
                        original_vllm_level = logging.getLogger("vllm").level
                        original_transformer_level = logging.getLogger("transformers").level
                        original_root_level = logging.getLogger().level
                        
                        # Set extremely strict logging levels
                        logging.getLogger("vllm").setLevel(logging.CRITICAL)
                        logging.getLogger("transformers").setLevel(logging.CRITICAL)
                        logging.getLogger().setLevel(logging.CRITICAL)
                        
                        try:
                            # Generate responses for the batch
                            batch_outputs = self.model.generate(
                                **batch_inputs,
                                max_new_tokens=5,  # Keep this small
                                min_new_tokens=1,  # Ensure at least something is generated
                                temperature=0.1,  # Using small non-zero temperature to avoid ValueError
                                return_dict_in_generate=True,
                                output_scores=True,
                                do_sample=True,  # Ensure we get some output
                                num_return_sequences=1
                            )
                            
                            # Process the generated responses
                            for i, (step_idx, input_len) in enumerate(zip(batch_indices, batch_inputs.input_ids.shape[1:])):
                                if i >= len(batch_outputs.sequences):
                                    step_scores[step_idx] = 0.5  # Default score if missing
                                    continue
                                    
                                # Extract the generated text for this prompt
                                response = self.tokenizer.decode(
                                    batch_outputs.sequences[i][input_len:], 
                                    skip_special_tokens=True
                                ).strip()
                                
                                # Try to extract a numerical score from the response
                                try:
                                    # Look for a number between 0 and 1
                                    import re
                                    numbers = re.findall(r"0\.\d+|\d+", response)
                                    if numbers:
                                        score = float(numbers[0])
                                        # Ensure score is between 0 and 1
                                        score = min(max(score, 0), 1)
                                    else:
                                        score = 0.5  # Default if no number found
                                except:
                                    score = 0.5  # Default if parsing fails
                                
                                step_scores[step_idx] = score
                                
                        except Exception as e:
                            print(f"Error during batch generation: {e}")
                            # Default scores already set
                        finally:
                            # Restore original logging levels
                            logging.getLogger("vllm").setLevel(original_vllm_level)
                            logging.getLogger("transformers").setLevel(original_transformer_level)
                            logging.getLogger().setLevel(original_root_level)
                
                question_scores.append(step_scores)
            
            all_scores.append(question_scores)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        return all_scores