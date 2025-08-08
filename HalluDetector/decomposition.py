import re
import os
from typing import Dict, List, Tuple
from openai import OpenAI
import time
import json
import logging

# Configuration
API_KEY      = "sk-df7855244c14491aa8a1bc656819fe50"
BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
WEB_MODEL    = "deepseek-v3"

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Prompt Template
SYSTEM_PROMPT_DECOUPLE = """
    ### **Instructions for Text Analysis and Extraction**

    You are a text analysis expert. Your task is to deconstruct a paragraph to isolate and extract its most valuable, substantive information. You will identify concrete factual observations and specific, actionable plans.

    #### **Core Process**

    Follow this three-step process for each paragraph:
    1.  **Fragment:** Split the text into micro-fragments, each containing a single, pure idea.
    2.  **Classify:** Label each fragment as either `observation` or `planning`.
    3.  **Extract:** Pull atomic claims from `observation` fragments and atomic actions from `planning` fragments.

    #### **Critical Filtering Principles**

    You must rigorously filter the text to extract **only definitive and concrete information**. The following content must be **strictly excluded**:

    * **Speculative or Hedged Language:** Any statement containing words like *may, might, could, possibly, likely, appears, seems, suggests*.
    * **Subjective Opinions:** Any evaluative language, such as *effective, ideal, best, good, useful*.
    * **Process Summaries & Meta-Commentary:** Do not extract comments about the search or work process. Extract the *results* of the work, not a summary of the effort.
        * **Incorrect Example:** "Progress has been made in identifying job opportunities."
        * **Correct Focus:** Extract the specific job opportunity itself (e.g., "OpenAI has a 'Research Scientist' role.").
    * **Vague Statements:** Process descriptions without specific results or plans without concrete steps.

    ---

    ### **Detailed Instructions**

    #### **1. Fragmentation**
    * Do not assume one sentence is one fragment. Sentences often contain multiple ideas.
    * Split sentences at clauses (e.g., around conjunctions like "but," "however," "and") to create **micro-fragments**.
    * The goal is to ensure each fragment is purely an observation or purely a plan. If a sentence contains both, create two separate fragments.

    #### **2. Classification**
    * **`observation`**: A fragment that presents specific, factual discoveries, concrete data, or definitive results.
    * **`planning`**: A fragment that describes a specific intended action, a concrete strategy, or a detailed future step.

    #### **3. Extraction (Rule of Atomicity)**
    From each fragment that passes the filtering principles, extract its core content as atomic items.

    * **Atomicity is Mandatory:** Every extracted claim or action must be a single, self-contained unit.
        * NO conjunctions (e.g., `and`, `or`, `but`).
        * NO conditionals (e.g., `if`, `when`, `unless`).
        * NO alternatives or choices.
        * Any statement containing these MUST be broken into separate atomic items.

    * **Crucial Example of Atomization:**
        * **INCORRECT (contains 'and'):** "Meta's careers page lists the role 'Research Scientist, Computer Vision' with locations in Menlo Park, CA, and Seattle, WA."
        * **CORRECT (broken into two atomic claims):**
            - "Meta's careers page lists the role 'Research Scientist, Computer Vision' with a location in Menlo Park, CA."
            - "Meta's careers page lists the role 'Research Scientist, Computer Vision' with a location in Seattle, WA."

    * **From `observation` Fragments, extract Atomic Claims:**
        * An atomic claim must be context-independent; its truthfulness can be assessed without the surrounding text.
        * State the verifiable fact or finding directly.
        * *Example*: If the fragment is "the data shows a 15% increase in user engagement," the claim is "User engagement increased by 15%."

    * **From `planning` Fragments, extract Atomic Actions:**
        * Identify the specific, concrete task to be performed.
        * Each action should define a clear objective, usually starting with a verb.
        * *Example*: If the fragment is "we will then deploy the model to the test server," the action is "Deploy the model to the test server."

    ---

    ### **Response Format**

    Only output fragments that contain extractable content. Your response must strictly adhere to the following format:

     ```
    Fragment 1: [micro-fragment text with substantive content]
    Classification: [observation/planning]
    Atomic [Claims/Actions]:
    - [substantive item 1]
    - [substantive item 2]
    ```

    If a paragraph contains no extractable substantive content, respond:
    ```
    No extractable content - paragraph contains only vague descriptions without specific results or concrete plans.
    ```
"""



SYSTEM_PROMPT_PURE_OBSERVATION_DECOMPOSITION = """
    ### **Instructions for Observation Extraction**

    You are a text analysis expert. Your task is to extract atomic claims from a paragraph that present specific, factual discoveries, concrete data, or definitive results.

    #### **Critical Filtering Principles**

    You must rigorously filter the text to extract **only definitive and concrete information**. The following content must be **strictly excluded**:

    * **Speculative or Hedged Language:** Any statement containing words like *may, might, could, possibly, likely, appears, seems, suggests*.
    * **Subjective Opinions:** Any evaluative language, such as *effective, ideal, best, good, useful*.
    * **Process Summaries & Meta-Commentary:** Do not extract comments about the search or work process. Extract the *results* of the work, not a summary of the effort.
        * **Incorrect Example:** "Progress has been made in identifying job opportunities."
        * **Correct Focus:** Extract the specific job opportunity itself (e.g., "OpenAI has a 'Research Scientist' role.").
    * **Vague Statements:** Process descriptions without specific results.
    * **Goals, Intentions, Difficulties, or Navigation/Meta Statements:** Such as "the goal is...", "still being sought", "an effective way...", "we plan", "we will", "plan to", "intend to", "aim to", "trying to", "attempt to".

    #### **Extraction Rules (Rule of Atomicity)**

    From each fragment that passes the filtering principles, extract its core content as atomic claims.

    * **Atomicity is Mandatory:** Every extracted claim must be a single, self-contained unit.
        * NO conjunctions (e.g., `and`, `or`, `but`).
        * NO conditionals (e.g., `if`, `when`, `unless`).
        * NO alternatives or choices.
        * Any statement containing these MUST be broken into separate atomic items.

    * **Crucial Example of Atomization:**
        * **INCORRECT (contains 'and'):** "Meta's careers page lists the role 'Research Scientist, Computer Vision' with locations in Menlo Park, CA, and Seattle, WA."
        * **CORRECT (broken into two atomic claims):**
            - "Meta's careers page lists the role 'Research Scientist, Computer Vision' with a location in Menlo Park, CA."
            - "Meta's careers page lists the role 'Research Scientist, Computer Vision' with a location in Seattle, WA."

    * **Extract Atomic Claims:**
        * An atomic claim must be context-independent; its truthfulness can be assessed without the surrounding text.
        * State the verifiable fact or finding directly.
        * Extract actual facts/information discovered, NOT that discovery happened.
        * *Example*: If the fragment is "the data shows a 15% increase in user engagement," the claim is "User engagement increased by 15%."

    #### **General Rules for Identifying Substantive Content**
    * Does it specify WHAT was found?
    * Does it contain concrete information (entities, attributes, locations, links, counts)?
    * Does it advance understanding or progress toward the query goal?
    * If it only states that work was done/will be done without specifics, exclude it.

    #### **URL/Link Handling**
    * If you find any URLs or links in the text, put them before their corresponding claim.
    * The "corresponding claim" is the text/sentence that appears BEFORE the URL in the original paragraph.
    * Do not modify the URLs at all - preserve them exactly as they appear in the source text.
    * URLs should be treated as additional information that supports or relates to the claim they precede.

    ---

    ### **Response Format**

    Only output atomic claims that contain extractable content. Your response must strictly adhere to the following format:

    ```
    - [URL/link 1] (if URL/link found in relation to item 1)
    - [substantive item 1]
    - [URL/link 2] (if URL/link found in relation to item 2)
    - [substantive item 2]
    ```

    If the paragraph contains no extractable substantive content, respond:
    ```
    No extractable content - paragraph contains only vague descriptions without specific results or concrete plans.
    ```
"""



SYSTEM_PROMPT_QUERY_DECOMPOSITION = """
    You are a helpful assistant specialized in extracting concise, 
    independent atomic constraints from a user's query. 
    An **atomic constraint** is single, self-contained, and indivisible. 
    Output only objective conditions or criteria—no personal references, background info, 
    or descriptive statements.
"""

USER_PROMPT_QUERY_DECOMPOSITION = """
    Extract all atomic constraint conditions from the following sentences. 
    If a sentence contains multiple claims 
    (e.g., linked by 'and', 'with', 'while'), you MUST break them down into separate atomic claims on new lines.\n\n"
    List each constraint on its own line prefixed with '- '. 
    Use neutral language—avoid pronouns like 'I', 'me', 'my', 'for me'. 
    Exclude any background or descriptive facts.\n\n"
    Input sentences:\n{sentences}\n\n"
    Provide only the list of constraints."
"""



SYSTEM_PROMPT_QUESTION_GENERATION = """You are a question generation expert. Your task is to convert atomic constraints into yes/no questions (general questions).

TASK REQUIREMENTS:
1. Convert each atomic constraint into a clear yes/no question
2. Questions should be designed to verify or investigate the constraint
3. Questions must be atomic (single, focused) and independent
4. Use "Does/Do/Is/Are" format to create yes/no questions
5. Ensure questions are answerable with yes or no

CONVERSION RULES:
- Convert statements into yes/no questions using "Does/Do/Is/Are"
- Maintain the core meaning and specificity of the original constraint
- Keep the same level of detail and precision

EXAMPLES:

Input constraints:
- The model achieves 95% accuracy on the test dataset


Output questions:
- Does the model achieve 95% accuracy on the test dataset?


RESPONSE FORMAT:
Output each question on a new line prefixed with "- ".
Example:
- [Question 1]
- [Question 2]
- [Question 3]
"""

# Helper: post-filter extracted items to enforce observation-only constraints
BANNED_HEDGES = (
    " may ", " might ", " could ", " possibly ", " likely ", " appears ", " seems ", " suggests ",
)
BANNED_PREFIXES = (
    "the goal is", "goal is", "an effective way", "effective way", "still being sought", "we plan", "we will", "plan to", "intend to", "aim to", "trying to", "attempt to"
)
BANNED_OPINION_WORDS = (
    "effective", "ideal", "best", "better", "worse"
)
BANNED_USER_WORDS = (
    "the user", "the requester", "the researcher", "the analyst", "the investigator"
)
BANNED_PLANNING_CUES = (
    " need to ", " should ", " will ", " plan to ", " intend to ", " look for ", " use ", " access ", " browse ", " search for "
)


def _is_valid_observation_item(text: str) -> bool:
    t = f" {text.strip().lower()} "
    if not text.strip():
        return False
    # filter hedges
    if any(h in t for h in BANNED_HEDGES):
        return False
    # filter meta/goal prefixes
    if any(t.strip().startswith(p) for p in BANNED_PREFIXES):
        return False
    # filter opinions
    if any(f" {w} " in t for w in BANNED_OPINION_WORDS):
        return False
    # filter user words
    if any(f" {w} " in t for w in BANNED_USER_WORDS):
        return False
    # filter planning cues (should never appear in observation claims)
    if any(cue in t for cue in BANNED_PLANNING_CUES):
        return False
    # filter statements about searching/navigation/process rather than facts
    if any(kw in t for kw in (" search ", " navigate ", " browsing ", " looking for ", " trying to find", " identified a page ")):
        return False
    return True

# --- URL helpers to prepend links to the corresponding claim_list ---
_URL_REGEX = re.compile(r'https?://[^\s)\]}>,]+')

def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = _URL_REGEX.findall(text)
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _prepend_urls(claim_list: List[str], urls: List[str]) -> None:
    if not urls:
        return
    # Insert at front preserving original order, avoiding duplicates already present
    for u in reversed(urls):
        if u not in claim_list:
            claim_list.insert(0, u)


def _extract_text_from_planning(planning_data) -> str:
    """Extract text content from planning data which may be string or dict."""
    if isinstance(planning_data, str):
        return planning_data
    elif isinstance(planning_data, dict):
        # Extract from title and description fields
        parts = []
        if 'title' in planning_data:
            parts.append(planning_data['title'])
        if 'description' in planning_data:
            parts.append(planning_data['description'])
        return ' '.join(parts)
    else:
        return str(planning_data) if planning_data else ""

# -------------- 1. Decouple planning and observation from reasoning text, then decompose them into atomic items -------------- #
def analyze_paragraph_fragments(paragraph, query):
    system_prompt = SYSTEM_PROMPT_DECOUPLE

    user_prompt = f"""Query: {query}

Paragraph to analyze: {paragraph}

Please classify this paragraph and decompose it into atomic claims or actions."""
    
    try:
        completion = client.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        
        response = completion.choices[0].message.content.strip()
        print(f"\nAnalyze Paragraph Response:\n{response}\n")
            
    except Exception as e:
        print(f"Error processing paragraph: {e}")
        return [], []

    # Parse the response, return two lists: observation_items and planning_items
    observation_items = []
    planning_items = []
    
    current_fragment = None
    current_classification = None
    current_items = []
    
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Check for fragment start
        if line.startswith("Fragment"):
            # Save previous fragment if exists
            if current_fragment and current_classification and current_items:
                if current_classification.lower() == "observation":
                    observation_items.extend(current_items)
                elif current_classification.lower() == "planning":
                    planning_items.extend(current_items)
            
            # Start new fragment
            current_fragment = line
            current_classification = None
            current_items = []
            
        # Check for classification
        elif line.startswith("Classification:"):
            current_classification = line.split(":", 1)[1].strip()
            
        # Check for atomic items
        elif line.startswith("Atomic") and ":" in line:
            # This line indicates the start of atomic items
            continue
            
        # Check for individual atomic items (lines starting with "-")
        elif line.startswith("- "):
            item = line[2:].strip()
            if item and not item.startswith("The user") and not item.startswith("The requester"):
                current_items.append(item)
    
    # Don't forget to save the last fragment
    if current_fragment and current_classification and current_items:
        if current_classification.lower() == "observation":
            observation_items.extend(current_items)
        elif current_classification.lower() == "planning":
            planning_items.extend(current_items)

    # Post-filter observation items strictly to observations without hedging/opinions/meta
    observation_items = [it for it in observation_items if _is_valid_observation_item(it)]

    print(f"\nExtracted Items:\nObservations: {observation_items}\nPlanning: {planning_items}\n")
    return observation_items, planning_items


# -------------- 2. Decompose the observation into atomic claims -------------- #
def decompose_observation(observation_paragraph, query):
    system_prompt = SYSTEM_PROMPT_PURE_OBSERVATION_DECOMPOSITION
    user_prompt = f"""Query: {query}

Paragraph to analyze: {observation_paragraph}

Please decompose this paragraph into atomic claims."""
    
    try:
        completion = client.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        response = completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error processing observation: {e}")
        return f"Error: {str(e)}"

    # Parse the response, return a list of atomic claims
    items = []
    for line in response.splitlines():
        if line.startswith("- "):
            txt = line[2:].strip()
            if _is_valid_observation_item(txt):
                items.append(txt)
    return items



# -------------- 3. Decompose the query into atomic constraints -------------- #
def decompose_query(query):
    system_prompt = SYSTEM_PROMPT_QUERY_DECOMPOSITION
    user_prompt = USER_PROMPT_QUERY_DECOMPOSITION.format(sentences=query)

    try:
        completion = client.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        
        response = completion.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error processing query: {e}")
        return f"Error: {str(e)}"

    items = []
    for line in response.splitlines():
        if line.startswith("- "):
            txt = line[2:].strip()
            # If the txt contains "the user" (insensitive to case), remove it
            if "the user" not in txt.lower():
                items.append(txt)
    return items


# -------------- 4. Convert atomic constraints into yes/no questions -------------- #
def get_question_via_LLM(constraints: List[str]) -> List[str]:
    """
    Convert atomic constraints into question-format atomic queries using LLM.
    
    Args:
        constraints: List of atomic constraints to convert
        
    Returns:
        List of question-format queries
    """
    if not constraints:
        return []
    
    system_prompt = SYSTEM_PROMPT_QUESTION_GENERATION
    user_prompt = f"""Convert the following atomic constraints into question-format queries:

Constraints:
{chr(10).join([f"- {constraint}" for constraint in constraints])}

Please generate one question for each constraint."""
    
    try:
        completion = client.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent output
        )
        
        response = completion.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []
    
    # Parse the response, return a list of questions
    questions = []
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("- "):
            question = line[2:].strip()
            if question:
                questions.append(question)
    
    return questions



def decompose_workflow_to_cache(input_json):
    data = json.load(open(input_json, 'r', encoding='utf-8'))
    chain = data.get('chain_of_research', {})
    
    # Determine the number of iterations by counting search steps
    num_iters = len([k for k in chain if k.startswith('search_')])
    query = data.get('query', '')
    
    # Initialize lists to store results
    iterations = []  # Will store ordered groups of action_list_N, search_list_N, claim_list_N
    
    print("-" * 60)

    # Decompose the query into sub-queries and always rebuild cache
    cache_file = f"json_cache/cache_{os.path.basename(input_json)}"
    print(f"Cache file: {cache_file}")
    if not os.path.exists(cache_file):
        print("Cache file not found, decomposing query...")
        sub_query_list = decompose_query(query)
       
        # Determine the pattern of the JSON file
        has_planning = any(k.startswith('plan_') for k in chain.keys())
        has_reasoning = any(k.startswith('reasoning_') for k in chain.keys())
        
        if has_planning and not has_reasoning:
            # Pattern 1: plan-search-observation
            print("Detected plan-search-observation pattern")
            
            # Preallocate iteration groups using the known searches
            iterations = [{
                'action_list': [],
                'search_list': chain.get(f'search_{idx}', []),
                'claim_list': []
            } for idx in range(1, num_iters + 1)]
            
            for i in range(1, num_iters + 1):
                print(f"\nProcessing Iteration {i}/{num_iters}...")
                
                # Get plan and observation for this iteration
                planning_text = chain.get(f'plan_{i}', '')
                observation_text = chain.get(f'observation_{i}', '')
                
                # Prepend any URLs from planning_text to the previous iteration's claim_list (if exists)
                if planning_text and i > 1:
                    planning_text_str = _extract_text_from_planning(planning_text)
                    _prepend_urls(iterations[i-2]['claim_list'], _extract_urls(planning_text_str))
                
                # Process planning text
                if planning_text:
                    print(f"\nProcessing planning text for iteration {i}:")
                    planning_text_str = _extract_text_from_planning(planning_text)
                    planning_obs_items, planning_plan_items = analyze_paragraph_fragments(planning_text_str, query)
                    # Add planning actions to current iteration action_list
                    iterations[i-1]['action_list'].extend(planning_plan_items)
                    # If observation items found in planning, add to previous claim_list
                    if planning_obs_items and i > 1:
                        iterations[i-2]['claim_list'].extend(planning_obs_items)
                
                # Process observation text
                if observation_text:
                    print(f"\nProcessing observation text for iteration {i}:")
                    # Check if observation contains "\n\n\n\n" and split accordingly
                    if "\n\n\n\n" in observation_text:
                        observation_parts = observation_text.split("\n\n\n\n")
                        for part in observation_parts:
                            if not part.strip():
                                continue
                            # Extract URLs from this specific part
                            part_urls = _extract_urls(part)
                            obs_items, plan_items = analyze_paragraph_fragments(part.strip(), query)
                            
                            # Insert URLs right before the claims from this part
                            if part_urls:
                                for url in part_urls:
                                    iterations[i-1]['claim_list'].append(url)
                            # Add claims to current iteration claim_list
                            iterations[i-1]['claim_list'].extend(obs_items)
                            
                            # If planning items found in observation, add to next action_list
                            if plan_items and i < num_iters:
                                iterations[i]['action_list'].extend(plan_items)
                    else:
                        # Prepend URLs from whole observation_text to current iteration's claim_list
                        _prepend_urls(iterations[i-1]['claim_list'], _extract_urls(observation_text))
                        obs_items, plan_items = analyze_paragraph_fragments(observation_text, query)
                        # Add claims to current iteration claim_list
                        iterations[i-1]['claim_list'].extend(obs_items)
                        # If planning items found in observation, add to next action_list
                        if plan_items and i < num_iters:
                            iterations[i]['action_list'].extend(plan_items)
                
                print(f"Finished Iteration {i}.\n" + "-" * 60)
                
        elif has_reasoning and not has_planning:
            # Pattern 2: reason-search
            print("Detected reason-search pattern")
            
            # Find all reasoning steps
            reasoning_steps = [k for k in chain.keys() if k.startswith('reasoning_')]
            reasoning_steps.sort(key=lambda x: int(x.split('_')[1]))
            
            # Initialize iterations list with empty groups for each search
            iterations = [{
                'action_list': [],
                'search_list': chain.get(f'search_{i}', []),
                'claim_list': []
            } for i in range(1, num_iters + 1)]
            
            for reasoning_key in reasoning_steps:
                reasoning_num = int(reasoning_key.split('_')[1])
                print(f"\nProcessing Reasoning {reasoning_num}...")
                
                reasoning_text = chain.get(reasoning_key, '')
                
                if reasoning_text:
                    print(f"\nProcessing reasoning text {reasoning_num}:")
                    # Determine target claim_list index for observations from this reasoning
                    target_idx = None
                    if reasoning_num > 1:
                        if reasoning_num - 2 < len(iterations):
                            target_idx = reasoning_num - 2
                        elif len(iterations) > 0:
                            target_idx = len(iterations) - 1
                    # Prepend any URLs found in reasoning_text to the target claim_list (only if obs would be placed)
                    if target_idx is not None:
                        _prepend_urls(iterations[target_idx]['claim_list'], _extract_urls(reasoning_text))

                    obs_items, plan_items = analyze_paragraph_fragments(reasoning_text, query)
                    
                    if reasoning_num == 1:
                        # For reasoning_1, only include planning_items in action_list_1
                        iterations[0]['action_list'].extend(plan_items)
                    else:
                        # For reasoning_x (x > 1): obs -> claim_list_(x-1)
                        if reasoning_num - 2 < len(iterations):
                            iterations[reasoning_num - 2]['claim_list'].extend(obs_items)
                        else:
                            # Extra reasoning after last search → add obs to final claim_list; ignore planning
                            if len(iterations) > 0:
                                iterations[-1]['claim_list'].extend(obs_items)
                        # plan -> action_list_x (if within search bounds)
                        if reasoning_num - 1 < len(iterations):
                            iterations[reasoning_num - 1]['action_list'].extend(plan_items)
                
                print(f"Finished Reasoning {reasoning_num}.\n" + "-" * 60)
        else:
            print("Error: Could not determine JSON pattern. Both planning and reasoning keys found or neither found.")
            return

        # Save the results to cache file
        cache_data = {
            'query_list': sub_query_list,
            'iterations': [
                {
                    f'action_list_{i+1}': group['action_list'],
                    f'search_list_{i+1}': group['search_list'],
                    f'claim_list_{i+1}': group['claim_list']
                }
                for i, group in enumerate(iterations)
            ]
        }
        
        # Ensure output directory exists if any
        _dir = os.path.dirname(cache_file)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
            # Modification?
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Results saved to {cache_file} (overwritten if existed)")
