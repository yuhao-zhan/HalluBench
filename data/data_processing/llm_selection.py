import pandas as pd
from openai import OpenAI
import json

# —— 配置 —— #
API_KEY      = "sk-df7855244c14491aa8a1bc656819fe50"
BASE_URL     = "https://dashscope.aliyuncs.com/compatible-mode/v1"
WEB_MODEL    = "qwen-max-latest"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def load_csv_data(csv_path):
    """Load CSV data and return the first 10 rows"""
    df = pd.read_csv(csv_path)
    first_10_rows = df.head(10)
    return first_10_rows

def build_mapping_relationship(data):
    """Build mapping relationship between task_id and task_description"""
    mapping = {}
    for _, row in data.iterrows():
        task_id = row['task_id']
        task_description = row['task_description']
        mapping[task_id] = task_description
    return mapping

def create_analysis_prompt(mapping):
    """Create a prompt for the LLM to analyze hallucination-prone tasks"""
    task_list = []
    for task_id, description in mapping.items():
        task_list.append(f"Task ID: {task_id}\nDescription: {description}\n")
    
    prompt = f"""You are an expert AI researcher analyzing the potential for hallucination in different types of tasks when using a deep research agent.

Below are 10 task descriptions that require web search and information gathering. Your task is to analyze which tasks are MORE prone to generating hallucinated or incorrect information when solved by a deep research agent.

Consider the following factors that make tasks more prone to hallucination:
1. **Complexity**: Tasks requiring multiple steps, cross-referencing, or synthesis of information
2. **Specificity**: Tasks requiring very specific, recent, or niche information
3. **Verification difficulty**: Tasks where it's hard to verify the accuracy of information
4. **Ambiguity**: Tasks with unclear requirements or multiple possible interpretations
5. **Dynamic content**: Tasks involving rapidly changing information (prices, availability, etc.)
6. **Multi-source synthesis**: Tasks requiring information from multiple sources that may conflict

Here are the 10 tasks to analyze:

{''.join(task_list)}

Based on your analysis, identify the 5 task IDs that are MOST prone to hallucination when using a deep research agent. 

Output ONLY the 5 task IDs in a comma-separated list, without any additional text or explanation.
"""

    return prompt

def analyze_with_llm(prompt):
    """Send prompt to LLM and get response"""
    try:
        response = client.chat.completions.create(
            model=WEB_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert AI researcher specializing in identifying tasks prone to hallucination in AI systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

def process_llm_response(response, mapping):
    """Process and validate the LLM response"""
    if not response:
        return None
    
    # Clean the response
    response = response.strip()
    
    # Split by comma and clean each task_id
    task_ids = [tid.strip() for tid in response.split(',')]
    
    # Validate that all task_ids exist in our mapping
    valid_task_ids = []
    invalid_task_ids = []
    
    for tid in task_ids:
        if tid in mapping:
            valid_task_ids.append(tid)
        else:
            invalid_task_ids.append(tid)
    
    if invalid_task_ids:
        print(f"Warning: Invalid task IDs found: {invalid_task_ids}")
    
    return valid_task_ids, invalid_task_ids

def main():
    # Load CSV data
    csv_path = "/data2/yuhaoz/DeepResearch/HalluBench/data/benchmark/opensource/mind2web_human_labeled.csv"
    data = load_csv_data(csv_path)
    
    # Build mapping relationship
    mapping = build_mapping_relationship(data)
    
    print("=== Task ID to Description Mapping ===")
    for task_id, description in mapping.items():
        print(f"\nTask ID: {task_id}")
        print(f"Description: {description}")
        print("-" * 80)
    
    # Create analysis prompt
    prompt = create_analysis_prompt(mapping)
    
    print("\n=== LLM Analysis Prompt ===")
    print(prompt)
    
    # Analyze with LLM
    print("\n=== LLM Analysis ===")
    result = analyze_with_llm(prompt)
    if result:
        print(f"LLM Response: {result}")
        
        # Process and validate the response
        valid_task_ids, invalid_task_ids = process_llm_response(result, mapping)
        
        if valid_task_ids:
            print(f"\n=== Valid Task IDs (Hallucination-Prone) ===")
            for i, tid in enumerate(valid_task_ids, 1):
                print(f"{i}. {tid}")
                print(f"   Description: {mapping[tid][:100]}...")
                print()
            
            # Save hallucination-prone task IDs to a separate file
            hallucination_tasks = {
                "total_tasks_analyzed": len(mapping),
                "hallucination_prone_tasks": len(valid_task_ids),
                "task_ids": valid_task_ids,
                "task_details": {tid: mapping[tid] for tid in valid_task_ids}
            }
            
            with open("hallucination_prone_tasks.json", "w", encoding="utf-8") as f:
                json.dump(hallucination_tasks, f, ensure_ascii=False, indent=2)
            print(f"Hallucination-prone task IDs saved to hallucination_prone_tasks.json")
        
        if invalid_task_ids:
            print(f"\n=== Invalid Task IDs ===")
            for tid in invalid_task_ids:
                print(f"- {tid}")
    else:
        print("Failed to get LLM response")
    
    # Save mapping to JSON for future use
    with open("task_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("\nTask mapping saved to task_mapping.json")

if __name__ == "__main__":
    main()
