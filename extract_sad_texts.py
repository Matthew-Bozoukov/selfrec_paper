import os
import json
import random
import re

min_text_len = 800#1200
max_text_len = 4000
full = True

def process_jsonl_files(directory_path):
    all_starts = []
    all_completions = []
    id_counter = 1 

    # Read each jsonl file and extract long content
    for filename in os.listdir(directory_path):
        if filename.endswith('.jsonl'):
            source = filename.split('.')[0]
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    data = json.loads(line)
                    content = data.get('content', '')
                    if len(content) > min_text_len * 2.1:
                        if full and len(content) > max_text_len: continue
                        start_part, remaining = split_content(content)
                        if remaining:
                            if full: second_part = remaining
                            else: second_part, _ = split_content(remaining)
                            all_starts.append({'id': f'id{id_counter}','source': source, 'text': start_part})
                            all_completions.append({'id': f'id{id_counter}','source': source, 'text': second_part})
                            id_counter += 1

    zipped_list = list(zip(all_starts, all_completions))
    random.shuffle(zipped_list)
    all_starts_shuffled, all_completions_shuffled = zip(*zipped_list)
    all_starts = list(all_starts_shuffled)
    all_completions = list(all_completions_shuffled)

    # Directories for outputs
    starts_dir = './starts_full'
    completions_dir = './completions_full'
    os.makedirs(starts_dir, exist_ok=True)
    os.makedirs(completions_dir, exist_ok=True)
    split_and_save_data(all_starts, starts_dir, 'starts')
    split_and_save_data(all_completions, completions_dir, 'completions_human')

def split_content(text):
    first_boundary = find_boundary(text, min_text_len)
    second_boundary = len(text) if full else find_boundary(text[first_boundary+1:], min_text_len) + first_boundary + 1
    return text[:first_boundary], text[first_boundary:second_boundary]

def find_boundary(text, char_limit):
    if len(text) <= char_limit:
        return len(text)
#    match = re.search(r'[\s\.\,\!\?\:\;\-\"\)\]\(\[\\]+', text[char_limit:])
    match = re.search(r'[a-zA-Z0-9]+([^a-zA-Z0-9])', text[char_limit:])
    if match:
        return char_limit + match.start(1)
    return len(text)

def split_and_save_data(data, directory, filename):
    train_size = 1000
    val_size = (len(data) - train_size) // 2
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    validation_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    for subset, name in zip([train_data, validation_data, test_data], ['train', 'validation', 'test']):
        path = os.path.join(directory, f'{filename}_{name}.json')
        with open(path, 'w') as f:
            json.dump(subset, f, indent=4)
        print(f"Saved {len(subset)} records to {path}")


if __name__ == "__main__":
    directory_path = './raw_sad_data'  
    process_jsonl_files(directory_path)
