import re
import os
from typing import Dict, List, Tuple
from openai import OpenAI
import time
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
from functools import partial

# Configuration
API_KEYS = [
    "sk-e3571f7b6c8f40d7bb22f3e24ed19fe5",
    "sk-462b0b0587f948c5ad814c6ef8d7b06d",
    "sk-bc332844ad9a433293231722a3bf7ee9",
    "sk-668c4d9427cc4d899c771656592e42c1",
    "sk-504f9bf2c9a1499699b3916ee3e18ff6",
    "sk-9fbe0e7ba4ba406591aedbdb556658ee",
    "sk-e7a77c525b3244719c0552a48a3d6ac4",
    "sk-fc1110e1047949218c21c6d49a5fee97",
    "sk-21ed35dcef0f47f38c1e48b0f5d7ec60",
    "sk-cf053eea342c425aa3ae312114b8a8f3",
    "sk-987aa24df5114e30bcf6c0e4993ddf82",
    "sk-b9bde29c16724157b7da326f58d51d73",
    "sk-df7855244c14491aa8a1bc656819fe50",
    "sk-81efa88acaf04a7b868d76e6811006af",
    "sk-5355487f104544c681c0aa9652418381",
    "sk-b90ea014cb204cb9804ff09e7d2bdf65",
    "sk-c4e08121bafb4abfa289b74c5088c6fe",
    "sk-c0fc17d4e8be4de6b2df939779baccab",
    "sk-b0d6018c1f604ca3a5fa911785d2c144",
    "sk-36402abb3f034d2eabadba38da507946"
]

# Legacy single API key for backward compatibility
API_KEY = API_KEYS[0]  # Keep the original key as first in list

BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
WEB_MODEL    = "qwen-max-latest"

# Parallel processing configuration
CPU_CORES = mp.cpu_count()  # Automatically detect available CPU cores
USE_PARALLEL = True  # Set to False to disable parallel processing
MAX_CONCURRENT_API_CALLS = min(20, CPU_CORES)  # Limit to available API keys

# Initialize OpenAI client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


def create_client_with_key(api_key: str) -> OpenAI:
    """
    Create a new OpenAI client with a specific API key.
    
    Args:
        api_key: The API key to use for this client
        
    Returns:
        OpenAI client instance
    """
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL
    )


def get_api_key_for_worker(worker_id: int) -> str:
    """
    Get an API key for a specific worker based on worker ID.
    This ensures even distribution of API keys across workers.
    
    Args:
        worker_id: The worker ID (0-based)
        
    Returns:
        API key string
    """
    return API_KEYS[worker_id % len(API_KEYS)]


def create_worker_client(worker_id: int) -> OpenAI:
    """
    Create a client for a specific worker with a dedicated API key.
    
    Args:
        worker_id: The worker ID (0-based)
        
    Returns:
        OpenAI client instance with worker-specific API key
    """
    api_key = get_api_key_for_worker(worker_id)
    return create_client_with_key(api_key)


def print_api_key_distribution():
    """
    Print the distribution of API keys across workers for monitoring.
    """
    print(f"\n🔑 API Key Distribution for Parallel Processing:")
    print(f"   Total API Keys: {len(API_KEYS)}")
    print(f"   Max Concurrent Workers: {MAX_CONCURRENT_API_CALLS}")
    print(f"   CPU Cores Available: {CPU_CORES}")
    
    print(f"\n   Worker ID → API Key (last 8 chars):")
    for worker_id in range(MAX_CONCURRENT_API_CALLS):
        api_key = get_api_key_for_worker(worker_id)
        short_key = api_key[-8:] if len(api_key) >= 8 else api_key
        print(f"   Worker {worker_id:2d} → ...{short_key}")
    
    print(f"\n   Note: Workers beyond {len(API_KEYS)} will cycle through the available keys")

# Prompt Template
SYSTEM_PROMPT_DECOUPLE = """
    ### **Instructions for Text Analysis and Extraction**

    You are a text analysis expert. Your task is to deconstruct a paragraph to isolate and extract its most valuable, substantive information. You will identify concrete factual observations and specific, actionable plans.

    #### **Core Process**

    Follow this three-step process for each paragraph:
    1.  **Fragment:** Split the text into micro-fragments, each containing a single, pure idea.
    2.  **Classify:** Label each fragment as either `observation` or `plan`.
    3.  **Extract:** Pull atomic claims from `observation` fragments and atomic actions from `plan` fragments.

    #### **Critical Filtering Principles**

    You must rigorously filter the text to extract **only definitive and concrete information**. The following content must be **strictly excluded**:

    * **Speculative or Hedged Language:** Any statement containing words like *may, might, could, possibly, likely, appears, seems, suggests*.
    * **Subjective Opinions:** Any evaluative language, such as *effective, ideal, best, good, useful*.
    * **Vague Statements:** Process descriptions without specific results or plans without concrete steps.

    ---

    ### **Detailed Instructions**

    #### **1. Fragmentation - CRITICAL RULE**
    * **NEVER assume one sentence = one fragment.**
    * **MANDATORY SPLITTING:** If a sentence contains BOTH observation AND plan, you MUST split it into separate fragments.
    * **Split at logical boundaries:** around conjunctions like "but," "however," "and," "while," "though."
    * **Each fragment must be PURE:** either 100% observation OR 100% plan, never mixed.
    * **Example:** "I found some roles, but I need to search more" → TWO fragments:
      - Fragment 1: "I found some roles" (observation)
      - Fragment 2: "I need to search more" (plan)

    #### **2. Classification**
    * **`observation`**: A fragment that presents specific, factual discoveries, concrete data, or definitive results.
    * **`plan`**: A fragment that describes a specific intended action, a concrete strategy, or a detailed future step.

    #### **3. Extraction (Rule of Atomicity)**
    From each fragment that passes the filtering principles, extract its core content as atomic items.

    * **Atomicity is Mandatory:** Every extracted claim or action must be a single, self-contained unit.
        * NO conjunctions (e.g., `and`, `or`, `but`).
        * NO conditionals (e.g., `if`, `when`, `unless`).
        * NO alternatives or choices.
        * Any statement containing these MUST be broken into separate atomic items.

    * **Crucial Example of Atomization:**
        * **INCORRECT (contains 'AND'):** "Meta's careers page lists the role 'Research Scientist, Computer Vision' with locations in Menlo Park, CA, and Seattle, WA."
        * **CORRECT (broken into two atomic claims):**
            - "Meta's careers page lists the role 'Research Scientist, Computer Vision' with a location in Menlo Park, CA."
            - "Meta's careers page lists the role 'Research Scientist, Computer Vision' with a location in Seattle, WA."

    * **From `observation` Fragments, extract Atomic Claims:**
        * An atomic claim must be context-independent; its truthfulness can be assessed without the surrounding text.
        * State the verifiable fact or finding directly.
        * **Process Summaries & Meta-Commentary:** Do not extract comments about the search or work process. Extract the *results* of the work, not a summary of the effort.
            * **Incorrect Example:** "Progress has been made in identifying job opportunities."
            * **Correct Focus:** Extract the specific job opportunity itself (e.g., "OpenAI has a 'Research Scientist' role.").
        * *Example*: If the fragment is "the data shows a 15% increase in user engagement," the claim is "User engagement increased by 15%."

    * **From `plan` Fragments, extract Atomic Actions:**
        * Identify the specific, concrete task to be performed.
        * Each action should define a clear objective, usually starting with a verb.
        * *Example*: If the fragment is "we will then deploy the model to the test server," the action is "Deploy the model to the test server."

    ---

    ### **Response Format**

    Only output fragments that contain extractable content. Your response must strictly adhere to the following format:

    **IMPORTANT:** Split mixed sentences into separate fragments!

    ```
    Fragment 1: [micro-fragment text with substantive content]
    Classification: [observation/plan]
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
You are a text analysis expert. Extract ONLY factual, concrete claims from the given text.

#### **What to EXCLUDE:**
- Speculative language (may, might, could, possibly, likely, appears, seems, suggests)
- Subjective opinions (effective, ideal, best, good, useful)
- Process summaries ("Progress has been made...", "We plan to...")
- Vague statements without specific results

#### **What to EXTRACT:**
- Specific facts, data, discoveries
- Concrete information (entities, locations, numbers, links)
- Results, not the process of finding them

#### **CRITICAL RULE - NO FAKE URLS:**
- ONLY extract URLs that are EXPLICITLY present in the source text, i.e., MUST be in the format of "https://xxx"
- NEVER invent, generate, or fabricate URLs
- NEVER add URLs that "make sense" for the topic
- NEVER add URLs that "should exist" for the subject
- NEVER convert entity names (like "Box Office Mojo", "Variety") into URLs
- A real URL must start with "http://" or "https://" and be explicitly written in the source
- If no URLs exist in the text, do NOT include any URL lines
- If you detect any URLs in the text, place URLs BEFORE their corresponding atomic claims. The "corresponding claim" is the factual statement that appears BEFORE the URL in the original text
- NEVER add placeholder text like "[URL]" or "[link]" when no URL exists

#### **Atomic Claims Rule:**
Break complex statements into single facts:
- ❌ "Meta has roles in Menlo Park AND Seattle"
- ✅ "Meta has a role in Menlo Park"
- ✅ "Meta has a role in Seattle"

#### **Response Format:**
If URLs exist in source text:
```
- [actual_URL_from_source] (if explicitly present in source text)
- [factual claim 1]
- [factual claim 2]
```

If NO URLs exist in source text:
```
- [factual claim 1]
- [factual claim 2]
```

If no extractable content: "No extractable content - paragraph contains only vague descriptions."
"""

SYSTEM_PROMPT_DOUBLE_CHECK_CLAIM = """You are a quality control expert. Review the claims and break down any non-atomic ones.

**Rules:**
- No conjunctions: and, or, but, while, though, however
- No conditionals: if, when, unless
- No compound statements

**Example:**
Input: "Role available in Menlo Park, CA and Seattle, WA"
Output: 
- Role available in Menlo Park, CA
- Role available in Seattle, WA

Return each atomic claim on a new line with "- " prefix."""


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

BANNED_USER_WORDS = (
    "the user", "the requester", "the researcher", "the analyst", "the investigator"
)



def _is_valid_observation_item(text: str) -> bool:
    t = f" {text.strip().lower()} "
    if not text.strip():
        return False
    # filter hedges
    
    if any(f" {w} " in t for w in BANNED_USER_WORDS):
        return False
    # filter planning cues (should never appear in observation claims)
    

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
    """Original function using the global client."""
    return analyze_paragraph_fragments_with_client(paragraph, query, client)


def analyze_paragraph_fragments_with_client(paragraph, query, client_instance):
    """
    Analyze paragraph fragments using a specific client instance.
    
    Args:
        paragraph: The paragraph text to analyze
        query: The query for context
        client_instance: OpenAI client instance to use
        
    Returns:
        Tuple of (observation_items, planning_items)
    """
    system_prompt = SYSTEM_PROMPT_DECOUPLE

    user_prompt = f"""Query: {query}

Paragraph to analyze: {paragraph}

Please classify this paragraph and decompose it into atomic claims or actions."""
    
    try:
        completion = client_instance.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent classification
        )
        
        response = completion.choices[0].message.content.strip()
        
        
            
    except Exception as e:
        print(f"Error processing paragraph: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
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
                print(f"Processing fragment: {current_fragment}")
                print(f"Classification: {current_classification}")
                print(f"Items: {current_items}")
                if current_classification.lower() in ["observation"]:
                    observation_items.extend(current_items)
                    print(f"Added {len(current_items)} observation items")
                elif current_classification.lower() in ["plan"]:
                    planning_items.extend(current_items)
                    print(f"Added {len(current_items)} planning items")
                else:
                    print(f"⚠️  Unknown classification: '{current_classification}' - skipping items")
            
            # Start new fragment
            current_fragment = line
            current_classification = None
            current_items = []
    
            
        # Check for classification
        elif line.startswith("Classification:"):
            current_classification = line.split(":", 1)[1].strip()
            
        # Check for atomic items (handle variations in format)
        elif line.startswith("Atomic") and ":" in line:
            # This line indicates the start of atomic items
            continue
        elif line.startswith("Atomic Claims:") or line.startswith("Atomic Actions:"):
            # Alternative format for atomic items
            continue
            
        # Check for individual atomic items (lines starting with "-")
        elif line.startswith("- "):
            item = line[2:].strip()
            current_items.append(item)
        elif line.startswith("-") and len(line.strip()) > 1:
            # Alternative format for atomic items
            item = line[1:].strip()
            current_items.append(item)
    
    # Don't forget to save the last fragment
    if current_fragment and current_classification and current_items:
        print(f"Processing final fragment: {current_fragment}")
        print(f"Final classification: {current_classification}")
        print(f"Final items:")
        for item in current_items:
            print(f"  - {item}")
        if current_classification.lower() in ["observation"]:
            observation_items.extend(current_items)
            print(f"Added final {len(current_items)} observation items")
        elif current_classification.lower() in ["plan"]:
            planning_items.extend(current_items)
            print(f"Added final {len(current_items)} planning items")
        else:
            print(f"⚠️  Unknown final classification: '{current_classification}' - skipping items")

    # Post-filter observation items strictly to observations without hedging/opinions/meta
    original_obs_count = len(observation_items)
    observation_items = [it for it in observation_items if _is_valid_observation_item(it)]
    filtered_obs_count = len(observation_items)
    if original_obs_count != filtered_obs_count:
        print(f"⚠️  Post-filtering: {original_obs_count - filtered_obs_count} observation items were filtered out")

    # Double-check observation items for non-atomic parts
    if observation_items:
        print(f"\n🔍 Double-checking {len(observation_items)} observation items for non-atomic parts...")
        observation_items = double_check_atomic_claims_with_client(observation_items, query, client_instance)
    
    return observation_items, planning_items


# -------------- 2. Decompose the observation into atomic claims -------------- #
def decompose_observation(observation_paragraph, query):
    """Original function using the global client."""
    return decompose_observation_with_client(observation_paragraph, query, client)


def decompose_observation_with_client(observation_paragraph, query, client_instance):
    """
    Decompose observation using a specific client instance.
    
    Args:
        observation_paragraph: The observation paragraph to decompose
        query: The query for context
        client_instance: OpenAI client instance to use
        
    Returns:
        List of atomic claims
    """
    system_prompt = SYSTEM_PROMPT_PURE_OBSERVATION_DECOMPOSITION
    user_prompt = f"""Query: {query}

Paragraph to analyze: {observation_paragraph}

Please decompose this paragraph into atomic claims."""
    
    try:
        completion = client_instance.chat.completions.create(
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
    
    # Double-check items for non-atomic parts
    print(f"Before double-check: {items}")
    for item in items:
        print(f"🔍 Debug: Item: {item}")
    if items:
        print(f"\n🔍 Double-checking {len(items)} observation items for non-atomic parts...")
        items = double_check_atomic_claims_with_client(items, query, client_instance)
    print(f"After double-check: {items}")   
    for item in items:
        print(f"🔍 Debug: Item: {item}")

    print()

    
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
            # If the txt contains BANNED_USER_WORDS (insensitive to case), remove it
            if not any(w in txt.lower() for w in BANNED_USER_WORDS):
                items.append(txt)

    # Turn the items into question format
    questions = get_question_via_LLM(items)
    return questions


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


# -------------- 5. Double-check atomic claims for non-atomic parts -------------- #
def double_check_atomic_claims(claims: List[str], query: str) -> List[str]:
    """Original function using the global client."""
    return double_check_atomic_claims_with_client(claims, query, client)


def double_check_atomic_claims_with_client(claims: List[str], query: str, client_instance: OpenAI) -> List[str]:
    """
    Double-check atomic claims to ensure they don't contain non-atomic parts like "and", "or".
    
    Args:
        claims: List of atomic claims to double-check
        query: The original query for context
        client_instance: OpenAI client instance to use
        
    Returns:
        List of filtered atomic claims that pass the double-check
    """
    if not claims:
        return []
    
    system_prompt = SYSTEM_PROMPT_DOUBLE_CHECK_CLAIM
    user_prompt = f"""Query: {query}

Please review and double-check the following atomic claims for any non-atomic elements:

Claims to review:
{chr(10).join([f"- {claim}" for claim in claims])}

Return only valid atomic claims, breaking down any non-atomic ones into separate atomic parts."""
    
    try:
        completion = client_instance.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent output
        )
        
        response = completion.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error double-checking atomic claims: {e}")
        # If double-check fails, return original claims
        return claims
    
    # Parse the response, return a list of double-checked claims
    filtered_claims = []
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("- "):
            claim = line[2:].strip()
            if claim and _is_valid_observation_item(claim):
                filtered_claims.append(claim)
    
    # If no claims were extracted from the response, return original claims
    if not filtered_claims:
        print(f"⚠️  Double-check returned no claims, using original {len(claims)} claims")
        return claims
    
    print(f"✅ Double-check completed: {len(claims)} → {len(filtered_claims)} claims")
    return filtered_claims


# -------------- 6. Parallel Processing Functions -------------- #

def process_paragraph_fragments_parallel(args_tuple: Tuple[int, str, str, int]) -> Tuple[int, List[str], List[str]]:
    """
    Process a single paragraph's fragments in parallel.
    
    Args:
        args_tuple: (paragraph_index, paragraph_text, query, worker_id)
        
    Returns:
        Tuple of (paragraph_index, observation_items, planning_items)
    """
    paragraph_index, paragraph_text, query, worker_id = args_tuple
    try:
        # Create worker-specific client with dedicated API key
        worker_client = create_worker_client(worker_id)
        api_key_short = get_api_key_for_worker(worker_id)[-8:]
        print(f"    Worker {worker_id} using API key ...{api_key_short} for paragraph {paragraph_index}")
        obs_items, plan_items = analyze_paragraph_fragments_with_client(paragraph_text, query, worker_client)
        return paragraph_index, obs_items, plan_items
    except Exception as e:
        print(f"❌ Error processing paragraph {paragraph_index}: {e}")
        return paragraph_index, [], []


def process_observation_parallel(args_tuple: Tuple[int, str, str, int]) -> Tuple[int, List[str]]:
    """
    Process a single observation paragraph in parallel.
    
    Args:
        args_tuple: (paragraph_index, paragraph_text, query, worker_id)
        
    Returns:
        Tuple of (paragraph_index, atomic_claims)
    """
    paragraph_index, paragraph_text, query, worker_id = args_tuple
    try:
        # Create worker-specific client with dedicated API key
        worker_client = create_worker_client(worker_id)
        api_key_short = get_api_key_for_worker(worker_id)[-8:]
        print(f"    Worker {worker_id} using API key ...{api_key_short} for paragraph {paragraph_index}")
        atomic_claims = decompose_observation_with_client(paragraph_text, query, worker_client)
        return paragraph_index, atomic_claims
    except Exception as e:
        print(f"❌ Error processing observation {paragraph_index}: {e}")
        return paragraph_index, []


def process_workflow_iteration_parallel(args_tuple: Tuple[int, Dict, str, str, int]) -> Tuple[int, Dict]:
    """
    Process a single workflow iteration in parallel.
    
    Args:
        args_tuple: (iteration_index, iteration_data, query, pattern_type, worker_id)
        
    Returns:
        Tuple of (iteration_index, processed_iteration_data)
    """
    iteration_index, iteration_data, query, pattern_type, worker_id = args_tuple
    
    try:
        # Create worker-specific client with dedicated API key
        worker_client = create_worker_client(worker_id)
        api_key_short = get_api_key_for_worker(worker_id)[-8:]
        print(f"    Worker {worker_id} using API key ...{api_key_short} for iteration {iteration_index}")
        
        if pattern_type == "planning":
            # Process planning and observation for this iteration
            planning_text = iteration_data.get('planning_text', '')
            observation_text = iteration_data.get('observation_text', '')
            
            action_list = []
            claim_list = []
            
            # Process planning text
            if planning_text:
                planning_text_str = _extract_text_from_planning(planning_text)
                planning_obs_items, planning_plan_items = analyze_paragraph_fragments_with_client(planning_text_str, query, worker_client)
                action_list.extend(planning_plan_items)
                # Note: planning observations need to be handled by caller for previous iteration
            
            # Process observation text
            if observation_text:
                if "\n\n\n\n" in observation_text:
                    observation_parts = observation_text.split("\n\n\n\n")
                    for part in observation_parts:
                        if not part.strip():
                            continue
                        part_urls = _extract_urls(part)
                        obs_items, plan_items = analyze_paragraph_fragments_with_client(part.strip(), query, worker_client)
                        
                        # Add URLs and claims
                        if part_urls:
                            claim_list.extend(part_urls)
                        claim_list.extend(obs_items)
                        action_list.extend(plan_items)
                else:
                    _prepend_urls(claim_list, _extract_urls(observation_text))
                    obs_items, plan_items = analyze_paragraph_fragments_with_client(observation_text, query, worker_client)
                    claim_list.extend(obs_items)
                    action_list.extend(plan_items)
            
            return iteration_index, {
                'action_list': action_list,
                'claim_list': claim_list,
                'planning_obs_items': planning_obs_items if planning_text else []
            }
            
        elif pattern_type == "reasoning":
            # Process reasoning for this iteration
            reasoning_text = iteration_data.get('reasoning_text', '')
            
            if reasoning_text:
                obs_items, plan_items = analyze_paragraph_fragments_with_client(reasoning_text, query, worker_client)
                return iteration_index, {
                    'observation_items': obs_items,
                    'planning_items': plan_items,
                    'urls': _extract_urls(reasoning_text)
                }
            else:
                return iteration_index, {
                    'observation_items': [],
                    'planning_items': [],
                    'urls': []
                }
        
        return iteration_index, {}
        
    except Exception as e:
        print(f"❌ Error processing iteration {iteration_index}: {e}")
        return iteration_index, {}


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
    cache_file = f"../json_cache/train_gemini/cache_{os.path.basename(input_json)}"
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
            
            if USE_PARALLEL and num_iters > 1:
                print(f"🚀 Using parallel processing with {CPU_CORES} CPU cores for {num_iters} iterations")
                print_api_key_distribution()
                
                # Prepare iteration data for parallel processing
                iteration_data = []
                for i in range(1, num_iters + 1):
                    iteration_data.append({
                        'iteration_index': i,
                        'planning_text': chain.get(f'plan_{i}', ''),
                        'observation_text': chain.get(f'observation_{i}', ''),
                        'search_list': chain.get(f'search_{i}', [])
                    })
                
                # Process iterations in parallel
                processed_iterations = [None] * num_iters
                
                with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_API_CALLS) as executor:
                    # Submit all iterations for parallel processing
                    future_to_iteration = {
                        executor.submit(process_workflow_iteration_parallel, 
                                     (data['iteration_index'], data, query, "planning", worker_id)): data['iteration_index']
                        for data, worker_id in zip(iteration_data, range(MAX_CONCURRENT_API_CALLS))
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_iteration):
                        iteration_index = future_to_iteration[future]
                        try:
                            idx, result = future.result()
                            processed_iterations[idx-1] = result
                            print(f"✅ Iteration {idx}/{num_iters} processed in parallel")
                        except Exception as e:
                            print(f"❌ Error processing iteration {iteration_index} in parallel: {e}")
                            # Fall back to sequential processing for this iteration
                            processed_iterations[iteration_index-1] = {
                                'action_list': [],
                                'claim_list': [],
                                'planning_obs_items': []
                            }
                
                # Build iterations list from parallel results
                iterations = []
                for i in range(num_iters):
                    if processed_iterations[i] is not None:
                        result = processed_iterations[i]
                        iterations.append({
                            'action_list': result.get('action_list', []),
                            'search_list': iteration_data[i]['search_list'],
                            'claim_list': result.get('claim_list', [])
                        })
                        
                        # Handle planning observations for previous iteration
                        if i > 0 and result.get('planning_obs_items'):
                            iterations[i-1]['claim_list'].extend(result['planning_obs_items'])
                    else:
                        # Fallback for failed iterations
                        iterations.append({
                            'action_list': [],
                            'search_list': iteration_data[i]['search_list'],
                            'claim_list': []
                        })
                
                print(f"🎉 Parallel workflow decomposition completed using {CPU_CORES} CPU cores")
                
            else:
                # Sequential processing (original logic)
                print(f"Using sequential processing for {num_iters} iterations")
                
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
            
            if USE_PARALLEL and len(reasoning_steps) > 1:
                print(f"🚀 Using parallel processing with {CPU_CORES} CPU cores for {len(reasoning_steps)} reasoning steps")
                print_api_key_distribution()
                
                # Prepare reasoning data for parallel processing
                reasoning_data = []
                for reasoning_key in reasoning_steps:
                    reasoning_num = int(reasoning_key.split('_')[1])
                    reasoning_data.append({
                        'iteration_index': reasoning_num,
                        'reasoning_text': chain.get(reasoning_key, ''),
                        'search_list': chain.get(f'search_{reasoning_num}', []) if reasoning_num <= num_iters else []
                    })
                
                # Process reasoning steps in parallel
                processed_reasoning = [None] * len(reasoning_steps)
                
                with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_API_CALLS) as executor:
                    # Submit all reasoning steps for parallel processing
                    future_to_reasoning = {
                        executor.submit(process_workflow_iteration_parallel, 
                                     (data['iteration_index'], data, query, "reasoning", worker_id)): data['iteration_index']
                        for data, worker_id in zip(reasoning_data, range(MAX_CONCURRENT_API_CALLS))
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_reasoning):
                        reasoning_index = future_to_reasoning[future]
                        try:
                            idx, result = future.result()
                            # Find the index in the reasoning_data list
                            data_idx = next(i for i, data in enumerate(reasoning_data) if data['iteration_index'] == idx)
                            processed_reasoning[data_idx] = result
                            print(f"✅ Reasoning {idx} processed in parallel")
                        except Exception as e:
                            print(f"❌ Error processing reasoning {reasoning_index} in parallel: {e}")
                            # Fall back to sequential processing for this reasoning step
                            data_idx = next(i for i, data in enumerate(reasoning_data) if data['iteration_index'] == reasoning_index)
                            processed_reasoning[data_idx] = {
                                'observation_items': [],
                                'planning_items': [],
                                'urls': []
                            }
                
                # Initialize iterations list with empty groups for each search
                iterations = [{
                    'action_list': [],
                    'search_list': chain.get(f'search_{i}', []),
                    'claim_list': []
                } for i in range(1, num_iters + 1)]
                
                # Build iterations from parallel reasoning results
                for i, (reasoning_data_item, processed_result) in enumerate(zip(reasoning_data, processed_reasoning)):
                    if processed_result is not None:
                        reasoning_num = reasoning_data_item['iteration_index']
                        
                        # Handle URLs
                        if processed_result.get('urls'):
                            target_idx = None
                            if reasoning_num > 1:
                                if reasoning_num - 2 < len(iterations):
                                    target_idx = reasoning_num - 2
                                elif len(iterations) > 0:
                                    target_idx = len(iterations) - 1
                            if target_idx is not None:
                                _prepend_urls(iterations[target_idx]['claim_list'], processed_result['urls'])
                        
                        # Handle observations and planning
                        obs_items = processed_result.get('observation_items', [])
                        plan_items = processed_result.get('planning_items', [])
                        
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
                
                print(f"🎉 Parallel reasoning processing completed using {CPU_CORES} CPU cores")
                
            else:
                # Sequential processing (original logic)
                print(f"Using sequential processing for {len(reasoning_steps)} reasoning steps")
                
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


def decompose_workflow_to_cache_auto(input_json):
    """
    Automatically choose between parallel and sequential workflow decomposition.
    
    Args:
        input_json: Path to the input JSON file
        
    Returns:
        None
    """
    if USE_PARALLEL:
        print(f"🚀 Auto-selecting parallel processing with {CPU_CORES} CPU cores")
        return decompose_workflow_to_cache(input_json)
    else:
        print(f"📝 Auto-selecting sequential processing")
        return decompose_workflow_to_cache(input_json)


def decompose_report_to_cache_auto(report: str, query: str, cache_file: str) -> bool:
    """
    Automatically choose between parallel and sequential report decomposition.
    
    Args:
        report: The full report text to decompose
        query: The original query for context
        cache_file: Path to the cache JSON file
        
    Returns:
        bool: True if decomposition was performed, False if skipped
    """
    if USE_PARALLEL:
        print(f"🚀 Auto-selecting parallel processing with {CPU_CORES} CPU cores")
        return decompose_report_to_cache_parallel(report, query, cache_file)
    else:
        print(f"📝 Auto-selecting sequential processing")
        return decompose_report_to_cache(report, query, cache_file)


def decompose_report_to_cache(report: str, query: str, cache_file: str) -> bool:
    """
    Decompose report paragraphs into atomic claims and save to cache file.
    
    Args:
        report: The full report text to decompose
        query: The original query for context
        cache_file: Path to the cache JSON file
        
    Returns:
        bool: True if decomposition was performed, False if skipped (already exists)
    """
    print(f"\n🔍 Checking if report decomposition already exists in cache...")
    
    # Check if cache file exists and has report attribute
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if report attribute already exists
            if 'report' in cache_data:
                print(f"✅ Report decomposition already exists in cache, skipping...")
                print(f"  Found {len(cache_data['report'])} paragraphs with atomic claims")
                return False
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️ Error reading cache file, will proceed with decomposition...")
    
    print(f"📝 Report decomposition not found, decomposing paragraphs...")
    
    # Split report into paragraphs
    paragraphs = [p.strip() for p in report.split('\n\n') if p.strip()]
    print(f"  Found {len(paragraphs)} paragraphs to decompose")
    
    # Decompose each paragraph into atomic claims
    report_data = []
    for i, paragraph in enumerate(paragraphs, 1):
        print(f"  Decomposing paragraph {i}/{len(paragraphs)}...")
        # print(f"    Paragraph: {paragraph}")
        try:
            # Decompose the paragraph into atomic claims
            atomic_claims = decompose_observation(paragraph, query)
            
            if not atomic_claims:
                print(f"    No extractable claims found in paragraph {i}")
                report_data.append({
                    'paragraph_index': i,
                    'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                    'atomic_claims': [],
                    'error': 'No extractable claims found'
                })
            else:
                print(f"    Extracted {len(atomic_claims)} claims from paragraph {i}")
                for claim in atomic_claims:
                    print(f"      - {claim}")
                report_data.append({
                    'paragraph_index': i,
                    'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                    'atomic_claims': atomic_claims
                })
                
        except Exception as e:
            print(f"    ❌ Error decomposing paragraph {i}: {str(e)}")
            report_data.append({
                'paragraph_index': i,
                'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                'atomic_claims': [],
                'error': str(e)
            })
    
    # Load existing cache data or create new
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            cache_data = {}
    else:
        cache_data = {}
    
    # Add report data to cache
    cache_data['report'] = report_data
    
    # Save updated cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Report decomposition saved to cache file: {cache_file}")
        print(f"  Saved {len(report_data)} paragraphs with atomic claims")
        return True
    except Exception as e:
        print(f"❌ Error saving report decomposition to cache: {str(e)}")
        return False


def decompose_report_to_cache_parallel(report: str, query: str, cache_file: str) -> bool:
    """
    Decompose report paragraphs into atomic claims in parallel and save to cache file.
    
    Args:
        report: The full report text to decompose
        query: The original query for context
        cache_file: Path to the cache JSON file
        
    Returns:
        bool: True if decomposition was performed, False if skipped (already exists)
    """
    print(f"\n🔍 Checking if report decomposition already exists in cache...")
    
    # Check if cache file exists and has report attribute
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if report attribute already exists
            if 'report' in cache_data:
                print(f"✅ Report decomposition already exists in cache, skipping...")
                print(f"  Found {len(cache_data['report'])} paragraphs with atomic claims")
                return False
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"⚠️ Error reading cache file, will proceed with decomposition...")
    
    print(f"📝 Report decomposition not found, decomposing paragraphs...")
    
    # Split report into paragraphs
    paragraphs = [p.strip() for p in report.split('\n\n') if p.strip()]
    print(f"  Found {len(paragraphs)} paragraphs to decompose")
    
    if not USE_PARALLEL or len(paragraphs) <= 1:
        print(f"  Using sequential processing for {len(paragraphs)} paragraphs")
        return decompose_report_to_cache(report, query, cache_file)
    
    print(f"  Using parallel processing with {CPU_CORES} CPU cores")
    print_api_key_distribution()
    
    # Prepare data for parallel processing
    paragraph_data = [(i, paragraph, query, i % MAX_CONCURRENT_API_CALLS) for i, paragraph in enumerate(paragraphs, 1)]
    
    # Process paragraphs in parallel
    report_data = [None] * len(paragraphs)  # Pre-allocate result array
    
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_API_CALLS) as executor:
        # Submit all paragraphs for parallel processing
        future_to_paragraph = {
            executor.submit(process_observation_parallel, data): data[0] 
            for data in paragraph_data
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_paragraph):
            paragraph_index = future_to_paragraph[future]
            try:
                idx, atomic_claims = future.result()
                # Convert to 0-based index for array
                array_idx = idx - 1
                
                if not atomic_claims:
                    print(f"    No extractable claims found in paragraph {idx}")
                    report_data[array_idx] = {
                        'paragraph_index': idx,
                        'paragraph_text': paragraphs[array_idx][:200] + "..." if len(paragraphs[array_idx]) > 200 else paragraphs[array_idx],
                        'atomic_claims': [],
                        'error': 'No extractable claims found'
                    }
                else:
                    print(f"    Extracted {len(atomic_claims)} claims from paragraph {idx}")
                    for claim in atomic_claims:
                        print(f"      - {claim}")
                    report_data[array_idx] = {
                        'paragraph_index': idx,
                        'paragraph_text': paragraphs[array_idx][:200] + "..." if len(paragraphs[array_idx]) > 200 else paragraphs[array_idx],
                        'atomic_claims': atomic_claims
                    }
                    
            except Exception as e:
                print(f"    ❌ Error processing paragraph {paragraph_index}: {str(e)}")
                array_idx = paragraph_index - 1
                report_data[array_idx] = {
                    'paragraph_index': paragraph_index,
                    'paragraph_text': paragraphs[array_idx][:200] + "..." if len(paragraphs[array_idx]) > 200 else paragraphs[array_idx],
                    'atomic_claims': [],
                    'error': str(e)
                }
    
    # Load existing cache data or create new
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            cache_data = {}
    else:
        cache_data = {}
    
    # Add report data to cache
    cache_data['report'] = report_data
    
    # Save updated cache
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Parallel report decomposition completed and saved to cache file: {cache_file}")
        print(f"  Saved {len(report_data)} paragraphs with atomic claims using {CPU_CORES} CPU cores")
        return True
    except Exception as e:
        print(f"❌ Error saving report decomposition to cache: {str(e)}")
        return False
