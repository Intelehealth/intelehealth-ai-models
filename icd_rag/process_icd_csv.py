#!/usr/bin/env python3

import csv
import json
import sys

def process_icd_csv(file_path):
    """
    Reads an ICD-11 CSV file, builds a hierarchical structure based on title dashes,
    groups by ChapterNo, and returns the structure as a dictionary.

    Args:
        file_path (str): The path to the ICD-11 CSV file.

    Returns:
        dict: A dictionary where keys are ChapterNo and values are lists of
              root nodes for that chapter, with nested children.
              Returns an empty dictionary if the file cannot be read or is empty.
    """
    icd_data = {}
    chapter_stacks = {} # {chapter_no: [(level, node)]}

    try:
        with open(file_path, mode='r', encoding='utf-8') as infile:
            # Handle potential Byte Order Mark (BOM)
            first_line = infile.readline()
            if first_line.startswith('\ufeff'):
                first_line = first_line[1:]

            # Combine the first line back with the rest for DictReader
            # Need to use an iterator chain or similar if file is huge,
            # but for typical CSVs, reading into memory is fine.
            # For simplicity here, we reset and let DictReader handle it.
            infile.seek(0)
            # Skip BOM again if present
            if infile.read(1) != '\ufeff':
                infile.seek(0)

            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                print(f"Error: Could not read header from {file_path}", file=sys.stderr)
                return {}

            # Clean field names (remove potential BOM or extra spaces)
            reader.fieldnames = [name.strip().lstrip('\ufeff') for name in reader.fieldnames]

            for row in reader:
                # Clean up potential extra columns if the header has a trailing comma
                row.pop(None, None)
                row.pop('', None)

                chapter_no_str = row.get('ChapterNo')
                if not chapter_no_str:
                    # Log or handle rows without ChapterNo if necessary
                    # print(f"Skipping row due to missing ChapterNo: {row.get('Code', 'N/A')}", file=sys.stderr)
                    continue

                # Map specific chapter strings and convert others to integer
                try:
                    if chapter_no_str == 'V':
                        chapter_no = 27
                    elif chapter_no_str == 'X':
                        chapter_no = 28
                    else:
                        # Convert other chapter numbers to integer for dictionary keys
                        chapter_no = int(chapter_no_str)
                except ValueError:
                    print(f"Warning: Could not convert ChapterNo '{chapter_no_str}' to integer. Skipping row {row.get('Code', 'N/A')}.", file=sys.stderr)
                    continue

                title = row.get('Title', '') # Get title, default to empty string if missing

                level = 0
                original_title = title
                # Count leading dashes precisely
                while title.startswith('-'):
                    level += 1
                    # Be careful with "--" vs "- -" - assuming contiguous dashes
                    if len(title) > 1 and title[1] == ' ': # Handle "- Title"
                        title = title[1:]
                        break
                    elif len(title) > 1 and title[1] == '-': # Handle "--Title"
                        title = title[2:] # Skip two chars for "--"
                    else: # Should be just "-"
                         title = title[1:]


                cleaned_title = title.strip()

                # Create node dictionary
                node = {key: value for key, value in row.items()} # Make a copy
                node['Title'] = cleaned_title
                # node['original_title'] = original_title # Optional: Keep original title
                node['level'] = level # Add level for clarity
                node['children'] = [] # Initialize children list for this node

                # Initialize chapter structure if first time seeing this chapter
                if chapter_no not in icd_data:
                    icd_data[chapter_no] = []
                    chapter_stacks[chapter_no] = []

                current_stack = chapter_stacks[chapter_no]

                # Find the correct parent level in the stack for this chapter
                while current_stack and current_stack[-1][0] >= level:
                    current_stack.pop()

                if current_stack:
                    # If stack is not empty, the last element is the parent
                    _parent_level, parent_node = current_stack[-1]
                    if parent_node and 'children' in parent_node:
                         parent_node['children'].append(node)
                    else:
                        # Fallback: Add as top-level if parent structure is broken
                        print(f"Warning: Could not find parent structure for {node.get('Code')}. Adding as top-level for chapter {chapter_no}.", file=sys.stderr)
                        icd_data[chapter_no].append(node)
                else:
                    # If stack is empty, this is a top-level node for this chapter
                    icd_data[chapter_no].append(node)

                # Push the current node onto the stack for its chapter
                current_stack.append((level, node))

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return {}

    return icd_data

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = '../intelehealth-datasets/icd_files/SimpleTabulation-ICD-11-MMS-en.csv'
    # Output JSON file path
    output_json_path = 'icd_11_mms_en_hierarchy.json'

    processed_data = process_icd_csv(csv_file_path)

    if processed_data:
        try:
            with open(output_json_path, 'w', encoding='utf-8') as outfile:
                json.dump(processed_data, outfile, indent=2, ensure_ascii=False)
            print(f"Successfully processed data and saved hierarchy to {output_json_path}")
        except IOError as e:
            print(f"Error writing JSON to file {output_json_path}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during JSON writing: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print("No data processed.", file=sys.stderr)
        sys.exit(1) 