#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import logging
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Any
import threading

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NLIActionChecker:
    """Singleton class for lazy-loading NLI model to prevent OOM in multiprocessing."""
    
    _instance = None
    _lock = threading.Lock()
    _model = None
    _tokenizer = None
    _device = None
    _model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def _initialize_model(self, force_cpu=False):
        """Lazy initialization of model with proper device management."""
        if self._model is None:
            try:
                # Determine device - prefer CPU to avoid GPU conflicts in multiprocessing
                if force_cpu or not torch.cuda.is_available():
                    self._device = torch.device("cpu")
                    print(f"🔧 Loading NLI model on CPU to avoid GPU conflicts")
                else:
                    # Check if GPU 0 has enough memory
                    try:
                        torch.cuda.set_device(0)
                        props = torch.cuda.get_device_properties(0)
                        allocated = torch.cuda.memory_allocated(0)
                        total = props.total_memory
                        available = total - allocated
                        
                        # Require at least 2GB free for the NLI model
                        if available < 2 * 1024**3:  # 2GB
                            print(f"⚠️ GPU 0 has only {available/1024**3:.2f}GB free, using CPU for NLI model")
                            self._device = torch.device("cpu")
                        else:
                            self._device = torch.device("cuda:0")
                            print(f"🔧 Loading NLI model on GPU 0 ({available/1024**3:.2f}GB available)")
                    except Exception as e:
                        print(f"⚠️ GPU check failed: {e}, using CPU for NLI model")
                        self._device = torch.device("cpu")
                
                # Load tokenizer and model
                print(f"📥 Loading {self._model_name} on {self._device}")
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
                self._model = self._model.to(self._device)
                
                print(f"✅ NLI model loaded successfully on {self._device}")
                
                # Memory check after loading
                if self._device.type == 'cuda':
                    allocated_after = torch.cuda.memory_allocated(0)
                    print(f"📊 GPU memory after NLI model loading: {allocated_after/1024**3:.2f}GB")
                
            except Exception as e:
                logger.error(f"❌ Failed to load NLI model: {e}")
                # Fallback to CPU if GPU fails
                if not force_cpu and self._device and self._device.type == 'cuda':
                    print("🔄 Retrying with CPU fallback...")
                    self._model = None
                    self._tokenizer = None
                    self._device = None
                    return self._initialize_model(force_cpu=True)
                else:
                    raise e
        
        return self._model, self._tokenizer, self._device
    
    def get_model_components(self):
        """Get model components with lazy loading."""
        return self._initialize_model()

# Global instance
_nli_checker = NLIActionChecker()

def nli_score(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Score the relationship between premise and hypothesis using NLI model.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        
    Returns:
        Dictionary with entailment, neutral, and contradiction scores
    """
    try:
        # Get model components with lazy loading
        model, tokenizer, device = _nli_checker.get_model_components()
        
        # Tokenize input
        input_text = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt", max_length=512)
        
        # Ensure all input tensors are on the same device as the model
        input_text = {k: v.to(device) for k, v in input_text.items()}
        
        # Get prediction
        with torch.no_grad():
            output = model(**input_text)
            prediction = torch.softmax(output["logits"][0], -1).tolist()
        
        # Map to labels
        label_names = ["entailment", "neutral", "contradiction"]
        scores = {name: float(pred) for pred, name in zip(prediction, label_names)}
        
        return scores
        
    except Exception as e:
        logger.error(f"❌ Error in NLI scoring: {e}")
        # Return neutral scores if error occurs
        return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

def split_text(text: str) -> List[str]:
    """
    Simple text splitting function to replace SummaC's split_text.
    Splits text into sentences based on common sentence endings.
    """
    # Simple sentence splitting - can be improved with more sophisticated NLP
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def check_actions_against_observations(query: str, atomic_observation_memory: str, atomic_actions: List[str]) -> List[Dict[str, Any]]:
    """
    Check each action against the observation memory using NLI scoring.
    
    Args:
        atomic_observation_memory: The observation memory paragraph
        atomic_actions: List of atomic actions to check (each action is a single sentence)
        
    Returns:
        List of dictionaries containing action judgments with scores and classification
    """
    if not atomic_actions:
        return []
    
    results = []
    
    print(f"🔍 Checking {len(atomic_actions)} actions against observation memory")
    # Concatenate the observation memory into a single string
    observation_memory = query + "\n\n" + atomic_observation_memory
    
    try:
        for j, action in enumerate(atomic_actions):
            if not action or not action.strip():
                continue
                
            # logger.info(f"Processing Action {j+1}: {action}")
            
            # Score the action directly against the observation memory
            scores = nli_score(observation_memory, action)
            
            # Determine the judgment (highest score among the three classifications)
            scores_dict = {
                "entailment": scores["entailment"],
                "contradiction": scores["contradiction"],
                "neutral": scores["neutral"]
            }
            
            judgment = max(scores_dict, key=scores_dict.get)
            
            result = {
                "action_idx": j,
                "action": action,
                "scores": {
                    "entailment": float(scores["entailment"]),
                    "contradiction": float(scores["contradiction"]),
                    "neutral": float(scores["neutral"])
                },
                "judgment": judgment,
                "judgment_score": float(scores_dict[judgment])
            }
            
            results.append(result)
            
            logger.info(f"    Judgment: {judgment} (Score: {scores_dict[judgment]:.3f})")
            
            # Memory cleanup for long lists
            if j % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        logger.error(f"❌ Error in action checking: {e}")
        # Return empty results if error occurs
        return []
    
    print(f"✅ Action checking completed: {len(results)} actions processed")
    return results

def cleanup_action_checker():
    """Clean up resources used by the action checker."""
    global _nli_checker
    if _nli_checker._model is not None:
        try:
            if _nli_checker._device and _nli_checker._device.type == 'cuda':
                torch.cuda.empty_cache()
            del _nli_checker._model
            del _nli_checker._tokenizer
            _nli_checker._model = None
            _nli_checker._tokenizer = None
            print("🧹 Action checker resources cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning up action checker: {e}")

# Usage example:
# 
# from action_checking import check_actions_against_observations
# 
# # Define your observation memory and actions
# observation_memory = "Your observation memory paragraph here..."
# actions = ["Action 1", "Action 2", "Action 3", ...]
# 
# # Check actions against observations
# judgments = check_actions_against_observations(observation_memory, actions)
# 
# # Access results
# for judgment in judgments:
#     print(f"Action {judgment['action_idx']}: {judgment['action']}")
#     print(f"Judgment: {judgment['judgment']} (Score: {judgment['judgment_score']:.3f})")
#     print(f"Scores - Entail: {judgment['scores']['entailment']:.3f}, "
#           f"Contra: {judgment['scores']['contradiction']:.3f}, "
#           f"Neutral: {judgment['scores']['neutral']:.3f}")
#
# # Clean up when done (optional)
# from action_checking import cleanup_action_checker
# cleanup_action_checker()