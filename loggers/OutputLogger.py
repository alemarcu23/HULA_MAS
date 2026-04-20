import os
import sys
import csv
import glob
import pathlib


def output_logger(fld, log_dir=None):
    log_base = log_dir if log_dir else os.path.join(fld, 'logs')
    if not os.path.isdir(log_base):
        print(f"No logs directory found at {log_base}")
        return

    # Find the most recent experiment folder inside logs/
    exp_dirs = glob.glob(os.path.join(log_base, 'exp_*/'))
    if not exp_dirs:
        print(f"No experiment folders found in {log_base}")
        return
    recent_dir = max(exp_dirs, key=os.path.getmtime)

    action_files = glob.glob(os.path.join(recent_dir, 'world_1/action*'))
    if not action_files:
        print(f"No action files found in {os.path.join(recent_dir, 'world_1')}")
        return
    action_file = action_files[0]

    action_header = []
    action_contents = []
    with open(action_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar="'")
        for row in reader:
            if not action_header:
                action_header = row
                continue
            res = {action_header[i]: row[i] for i in range(len(action_header))}
            action_contents.append(res)

    if not action_contents:
        print("Action file is empty")
        return

    # Dynamically identify agent and human columns from the header.
    # Columns follow the pattern: <entity_id>_action, <entity_id>_location
    agent_action_cols = [h for h in action_header if h.endswith('_action') and 'human' not in h.lower()]
    human_action_cols = [h for h in action_header if h.endswith('_action') and 'human' in h.lower()]

    unique_agent_actions = set()
    unique_human_actions = set()
    for entry in action_contents:
        for col in agent_action_cols:
            val = entry.get(col, '')
            if val:
                loc_col = col.replace('_action', '_location')
                unique_agent_actions.add((val, entry.get(loc_col, '')))
        for col in human_action_cols:
            val = entry.get(col, '')
            if val:
                loc_col = col.replace('_action', '_location')
                unique_human_actions.add((val, entry.get(loc_col, '')))
        # Count cooperative human actions as agent actions too
        for col in human_action_cols:
            val = entry.get(col, '')
            if val in ('RemoveObjectTogether', 'CarryObjectTogether', 'DropObjectTogether'):
                loc_col = col.replace('_action', '_location')
                unique_agent_actions.add((val, entry.get(loc_col, '')))

    # Retrieve the number of ticks to finish the task, score, and completeness
    no_ticks = action_contents[-1].get('tick_nr', '')
    score = action_contents[-1].get('score', '')
    completeness = action_contents[-1].get('completeness', '')

    # Save the output as a csv file
    print("Saving output...")
    output_path = os.path.join(recent_dir, 'world_1', 'output.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['completeness', 'score', 'no_ticks', 'agent_actions', 'human_actions'])
        csv_writer.writerow([completeness, score, no_ticks, len(unique_agent_actions), len(unique_human_actions)])
