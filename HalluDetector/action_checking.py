#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import logging
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Any

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLI model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)

def nli_score(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Score the relationship between premise and hypothesis using NLI model.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        
    Returns:
        Dictionary with entailment, neutral, and contradiction scores
    """
    input_text = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    input_text = {k: v.to(device) for k, v in input_text.items()}
    
    with torch.no_grad():
        output = model(**input_text)
        prediction = torch.softmax(output["logits"][0], -1).tolist()
    
    label_names = ["entailment", "neutral", "contradiction"]
    scores = {name: float(pred) for pred, name in zip(prediction, label_names)}
    
    return scores

def split_text(text: str) -> List[str]:
    """
    Simple text splitting function to replace SummaC's split_text.
    Splits text into sentences based on common sentence endings.
    """
    # Simple sentence splitting - can be improved with more sophisticated NLP
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def check_actions_against_observations(atomic_observation_memory: str, atomic_actions: List[str]) -> List[Dict[str, Any]]:
    """
    Check each action against the observation memory using NLI scoring.
    
    Args:
        atomic_observation_memory: The observation memory paragraph
        atomic_actions: List of atomic actions to check (each action is a single sentence)
        
    Returns:
        List of dictionaries containing action judgments with scores and classification
    """
    results = []
    
    for j, action in enumerate(atomic_actions):
        logger.info(f"Processing Action {j+1}: {action}")
        
        # Score the action directly against the observation memory
        scores = nli_score(atomic_observation_memory, action)
        
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
    
    return results

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