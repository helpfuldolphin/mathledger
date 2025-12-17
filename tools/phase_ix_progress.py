import re
import json
import os

def parse_progress(file_path):
    """
    Parses a markdown file to count checked, unchecked, and in-progress task list items.

    Args:
        file_path (str): The path to the markdown file.

    Returns:
        dict: A dictionary with counts for 'TODO', 'IN_PROGRESS', and 'DONE'.
              Returns an error message if the file is not found.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {"error": f"Failed to read file: {e}"}

    # Regex to find markdown checkboxes
    unchecked_pattern = r'-\s\[\s\]'
    in_progress_pattern = r'-\s\[~\]'
    checked_pattern = r'-\s\[[xX]\]'

    todo_count = len(re.findall(unchecked_pattern, content))
    in_progress_count = len(re.findall(in_progress_pattern, content))
    done_count = len(re.findall(checked_pattern, content))

    return {
        "TODO": todo_count,
        "IN_PROGRESS": in_progress_count,
        "DONE": done_count
    }

if __name__ == '__main__':
    # Assume the script is in 'tools/' and the markdown is in 'docs/phase_ix/'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    markdown_file = os.path.join(project_root, 'docs', 'phase_ix', 'Phase_IX_PR_Plan.md')
    
    progress_data = parse_progress(markdown_file)
    
    print(json.dumps(progress_data, indent=4))
