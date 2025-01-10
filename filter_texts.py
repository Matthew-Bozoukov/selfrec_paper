from data import *

def filter_to_equal_len(filterfile, targetfile, outfile1, outfile2, tolerance = .52):
    texts_to_filter = load_from_json(filterfile)
    target_texts = load_from_json(targetfile)
    filtered_texts1 = {}
    filtered_texts2 = {}
    meanlendiff, meanlendiffpct = 0, 0
    for k, v in texts_to_filter.items():
        if len(v) >= len(target_texts[k]) * (1 - tolerance) and len(v) <= len(target_texts[k]) * (1 + tolerance):
            filtered_texts1[k] = v
            filtered_texts2[k] = target_texts[k]
            meanlendiff += len(v) - len(target_texts[k])
            meanlendiffpct += (len(v) - len(target_texts[k])) / len(target_texts[k])
    meanlendiff /= len(filtered_texts1)
    print(f"Mean length difference: {meanlendiff}")
    meanlendiffpct /= len(filtered_texts1)
    print(f"Mean length difference percentage: {meanlendiffpct}")
    print(f"Filtered {len(filtered_texts1)} texts")
    with open(outfile1, 'w') as f:
        json.dump(filtered_texts1, f, indent=4)
    with open(outfile2, 'w') as f:
        json.dump(filtered_texts2, f, indent=4)
            

if __name__ == "__main__":
    filter_to_equal_len("summaries/dolly_train_llama3_8bchat_responses.json", "summaries/dolly_train_human_responses.json",
                        "summaries/dolly_train_llama3_8bchat_filteredlen_responses.json", "summaries/dolly_train_human_filteredlen_responses.json")