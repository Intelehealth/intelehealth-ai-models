import pandas as pd
import re
import sys
import json

def extract_numbered_diagnoses(llm_diagnosis_text):
    """Extract numbered diagnoses from LLM Diagnosis text."""
    diagnoses = []
    if pd.isna(llm_diagnosis_text):
        return diagnoses
    
    # Check if the text is a space-separated list with numbered items (e.g., "Mastitis 2. Breast Abscess")
    space_numbered_pattern = r'(\w+(?:\s+\w+)*)\s+(\d+)\.\s+'
    if re.search(space_numbered_pattern, llm_diagnosis_text):
        # This is a space-separated list with numbers
        # First, extract the first diagnosis (which might not have a number)
        first_diagnosis_match = re.match(r'^([^0-9]+?)(?=\s+\d+\.|\s*$)', llm_diagnosis_text)
        if first_diagnosis_match:
            first_diagnosis = first_diagnosis_match.group(1).strip()
            if first_diagnosis:
                diagnoses.append((1, first_diagnosis))
        
        # Then extract the numbered diagnoses
        numbered_items = re.findall(r'(\d+)\.\s+([^0-9]+?)(?=\s+\d+\.|\s*$)', llm_diagnosis_text)
        for number_str, diagnosis in numbered_items:
            number = int(number_str)
            if number <= 5:
                diagnoses.append((number, diagnosis.strip()))
        
        return [d[1] for d in sorted(diagnoses)]
    
    # First, check if the text contains numbered diagnoses (e.g., "2. Typhoid Fever")
    numbered_pattern = r'(?:\*\*)?(\d+)\.(?:\*\*)?\s+([^*\n,]+?)(?:\s+\(|\*\*|\n|$|,)'
    numbered_matches = list(re.finditer(numbered_pattern, llm_diagnosis_text))
    
    if numbered_matches:
        # Process numbered diagnoses
        for match in numbered_matches:
            number = int(match.group(1))
            diagnosis = match.group(2).strip()
            # Ensure we only get the first 5 diagnoses
            if number <= 5:
                # Split by slashes if present
                sub_diagnoses = [d.strip() for d in re.split(r'[/]', diagnosis)]
                for sub_diagnosis in sub_diagnoses:
                    if sub_diagnosis:
                        diagnoses.append((number, sub_diagnosis))
    else:
        # If no numbered diagnoses found, try to extract diagnoses from a comma-separated list
        # Split by commas and process each item
        items = [item.strip() for item in llm_diagnosis_text.split(',')]
        for i, item in enumerate(items):
            # Check if the item contains a number followed by a period
            number_match = re.match(r'(?:\*\*)?(\d+)\.(?:\*\*)?\s+(.*)', item)
            if number_match:
                number = int(number_match.group(1))
                diagnosis = number_match.group(2).strip()
            else:
                # If no number found, assign a sequential number
                number = i + 1
                diagnosis = item
            
            # Split by slashes if present
            sub_diagnoses = [d.strip() for d in re.split(r'[/]', diagnosis)]
            for sub_diagnosis in sub_diagnoses:
                if sub_diagnosis:
                    diagnoses.append((number, sub_diagnosis))
    
    # Sort by number and return just the diagnoses
    return [d[1] for d in sorted(diagnoses)]

def create_diagnosis_mapping(csv_file):
    """Create a dictionary mapping diagnoses to their matching LLM diagnoses."""
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get total number of rows in the CSV
    total_rows = len(df)
    
    # Initialize the mapping dictionary and total top_1_hits
    diagnosis_mapping = {}
    total_top_1_hits = 0
    # Initialize counters for top_5_hits positions
    top_5_hits_counts = {"1/5": 0, "2/5": 0, "3/5": 0, "4/5": 0, "5/5": 0}

    for _, row in df.iterrows():
        if pd.isna(row['Diagnosis']) or pd.isna(row['LLM Diagnosis']):
            continue
            
        # Get the ground truth diagnosis (remove any text after colon)
        gt_diagnosis = row['Diagnosis'].split(':')[0].strip()
        
        # Extract LLM diagnoses
        llm_diagnoses = extract_numbered_diagnoses(row['LLM Diagnosis'])
        
        # Get top 1 and top 5 hit values
        top_1_hit = row.get('Top 1 ddx hit', 0)
        top_5_hit = row.get('Top 5 ddx hit', "0/5")
        
        # Initialize the mapping entry if it doesn't exist
        if gt_diagnosis not in diagnosis_mapping:
            diagnosis_mapping[gt_diagnosis] = {
                'matched_llm_diagnoses': [],
                'top_1_hits': 0,
                'top_5_hits': []
            }
        
        # If we have a top 1 hit
        if str(top_1_hit) == "1" and llm_diagnoses:
            # Ensure no duplicates are added
            if llm_diagnoses[0] not in diagnosis_mapping[gt_diagnosis]['matched_llm_diagnoses']:
                diagnosis_mapping[gt_diagnosis]['matched_llm_diagnoses'].append(llm_diagnoses[0])
            diagnosis_mapping[gt_diagnosis]['top_1_hits'] += 1
            total_top_1_hits += 1
            # When we have a top 1 hit, it's also a 1/5 hit
            top_5_hits_counts["1/5"] += 1
            if "1/5" not in diagnosis_mapping[gt_diagnosis]['top_5_hits']:
                diagnosis_mapping[gt_diagnosis]['top_5_hits'].append("1/5")
        
        # If we have a top 5 hit that's not a top 1 hit
        elif isinstance(top_5_hit, str) and '/' in top_5_hit:
            try:
                rank = int(top_5_hit.split('/')[0])
                if 1 <= rank <= 5 and len(llm_diagnoses) >= rank:
                    # Ensure no duplicates are added
                    if llm_diagnoses[rank-1] not in diagnosis_mapping[gt_diagnosis]['matched_llm_diagnoses']:
                        diagnosis_mapping[gt_diagnosis]['matched_llm_diagnoses'].append(llm_diagnoses[rank-1])
                    
                    hit_position = f"{rank}/5"
                    if hit_position not in diagnosis_mapping[gt_diagnosis]['top_5_hits']:
                        diagnosis_mapping[gt_diagnosis]['top_5_hits'].append(hit_position)
                    
                    # Increment the counter for this rank
                    top_5_hits_counts[hit_position] += 1
            except (ValueError, IndexError):
                # Skip if there's an issue parsing the top_5_hit value
                pass
    
    # Calculate total top 5 hits - this should include all hits from rank 1-5
    total_top_5_hits = sum(top_5_hits_counts.values())
    
    print(f"Total Rows in CSV: {total_rows}")
    print(f"Total Top 1 Hits: {total_top_1_hits}")
    print(f"Total Top 5 Hits: {total_top_5_hits}")
    # Print counts for each top_5_hits position
    print("Top 5 Hits Counts:")
    for rank, count in top_5_hits_counts.items():
        print(f"  {rank}: {count}")
    
    return diagnosis_mapping

def print_mapping_summary(mapping):
    """Print a summary of the diagnosis mapping."""
    print("\nDiagnosis Mapping Summary:")
    print("-" * 80)
    
    for gt_diagnosis, data in mapping.items():
        print(f"\nGround Truth Diagnosis: {gt_diagnosis}")
        print(f"Total Top 1 Hits: {data['top_1_hits']}")
        print(f"Top 5 Hit Positions: {', '.join(data['top_5_hits']) if data['top_5_hits'] else 'None'}")
        print("Matched LLM Diagnoses:")
        for diagnosis in set(data['matched_llm_diagnoses']):  # Using set to remove duplicates
            print(f"  - {diagnosis}")
        print("-" * 40)

if __name__ == "__main__":
    # Use the most recent CSV file
    csv_file = "./data/v2_results/gemini_2_flash_nas_combined_ayu_inference_final.csv"
    
    # Create the mapping
    mapping = create_diagnosis_mapping(csv_file)
    
    # Dump the mapping to a JSON file
    with open('diagnosis_mapping.json', 'w') as json_file:
        json.dump(mapping, json_file, indent=4)
    
    print(mapping)
    # Print the summary
    # print_mapping_summary(mapping) 
    
    # Load the diagnosis mapping from the JSON file
    with open('diagnosis_mapping.json', 'r') as json_file:
        mapping = json.load(json_file)

    # Find and print diagnoses with top_1_hits as 1
    for gt_diagnosis, data in mapping.items():
        if data['top_1_hits'] == 1:
            print(f"Ground Truth Diagnosis: {gt_diagnosis} has top_1_hits as 1.") 
            print("Matched LLM Diagnoses:")
            for diagnosis in data['matched_llm_diagnoses']:
                print(f"  - {diagnosis}")
            print("-" * 40)
    