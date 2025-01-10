import json
from datasets import load_dataset
import os

SOURCES = ["claude", "gpt35", "gpt4", "llama", "llama2_13bchat", "llama3_8bchat", "llama3_8bbase", "sonnet", "human_filteredlen", "llama3_8bchat_filteredlen", "human"]#


def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)

def write_dolly_articles_and_human_summaries():
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dolly_sum = dataset.filter(lambda x: x["category"] == "summarization")
    sumdict = {}
    articles = {}
    instructions = {}
    for i in range(len(dolly_sum)):
        sumdict[f"id{i}"] = dolly_sum[i]['response'].replace("\n\n", "\n")
        articles[f"id{i}"] = dolly_sum[i]['context'].replace("\n\n", "\n")
        instructions[f"id{i}"] = dolly_sum[i]['instruction']
    save_to_json(sumdict, f"summaries/dolly_train_human_responses.json")
    save_to_json(articles, f"articles/dolly_train_articles.json")
    save_to_json(instructions, f"articles/dolly_train_instructions.json")
        
def write_articles_and_human_summaries(ds_name = "cnn", split = "train", start = 0, n = 1000):
    if ds_name == "cnn":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        article_name = 'article'
        summary_name = 'highlights'
    else:
        dataset = load_dataset("EdinburghNLP/xsum", split=split)
        article_name = 'document'
        summary_name = 'summary'
    sumdict = {}
    articles = {}
    for i in range(start, min(start+n,len(dataset))):
        sumdict[dataset[i]['id']] = dataset[i][summary_name].replace(" .\n", "\n") ## for CNN highlights
        if sumdict[dataset[i]['id']][-2:] == " .": sumdict[dataset[i]['id']] = sumdict[dataset[i]['id']][:-2]
        articles[dataset[i]['id']] = dataset[i][article_name]
    save_to_json(sumdict, f"summaries/{ds_name}_{split}{start}to{i}_human_responses.json")
    save_to_json(articles, f"articles/{ds_name}_{split}{start}to{i}_articles.json")
    
def memoize(func):
    cache = {}
    def memoized_func(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return memoized_func

@memoize
def load_data(dataset):
    responses = {}
    for source in SOURCES:
        # first check if file exists
        if not os.path.exists(f"summaries/{dataset}_train_{source}_responses.json"):
            continue
        responses[source] = load_from_json(
            f"summaries/{dataset}_train_{source}_responses.json"
        )
    articles = load_from_json(f"articles/{dataset}_train_articles.json")
    keys = list(responses[source].keys())#keys = list(articles.keys())
    return responses, articles, keys

def load_data_sad(ddir = "completions"):
    responses = {}
    for source in SOURCES:
        # first check if file exists
        fname = f"{ddir}/completions_{source}_train.json"
        if not os.path.exists(fname):
            continue
        completions = load_from_json(fname)
        responses[source] = {d['id']: d['text'] for d in completions}
        keys = list(responses[source].keys())
#    keys = list({inner_dict['id'] for response_list in responses.values() for inner_dict in response_list})
    return responses, keys
    
def initial_load_data_dolly():
    articles = load_from_json(f"articles/dolly_train_articles.json")
    keys = list(articles.keys())
    instructions = {}
    instructions = load_from_json(f"articles/dolly_train_instructions.json")
    return articles, instructions, keys

@memoize
def load_data_dolly():
    responses = {}
    for source in SOURCES:
        # first check if file exists
        if not os.path.exists(f"summaries/dolly_train_{source}_responses.json"):
            continue
        responses[source] = load_from_json(
            f"summaries/dolly_train_{source}_responses.json"
        )
    articles = load_from_json(f"articles/dolly_train_articles.json")
    keys = list(responses[source].keys())#keys = list(articles.keys())
    instructions = load_from_json(f"articles/dolly_train_instructions.json")
    return responses, articles, instructions, keys
    
def load_cnn_dailymail_data():
    """
    cnn_train: 287113 items
    cnn_test: 11490 items
    cnn_validation: 13368 items

    article: ~781 tokens
    highlights: ~56 tokens
    id
    """
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]

    return train_data, test_data, validation_data


def load_xsum_data():
    """
    xsum_train: ~204045 items
    xsum_test: ~11334 items
    xsum_validation: ~11332 items

    document: ~2200 chars
    summary: ~125 chairs
    id
    """
    dataset = load_dataset("EdinburghNLP/xsum")

    train_data = dataset["train"]
    test_data = dataset["test"]
    validation_data = dataset["validation"]

    return train_data, test_data, validation_data


def write_to_jsonl_for_finetuning(
    questions, answers, system_prompt, file_name="finetuningdata.jsonl"
):
    formatted_data = ""

    for question, answer in zip(questions, answers):
        entry = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
        formatted_data += json.dumps(entry) + "\n"

    with open(file_name, "w") as file:
        file.write(formatted_data)
