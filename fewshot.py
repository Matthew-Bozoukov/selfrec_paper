### Functions to make fewshot prompts for base models

from prompts import *
from data import load_data, load_data_dolly
from initialize import *

def make_fewshot_summaries_prompt(dataset, new_article, new_instruction = ""):
    models = ["human", "claude", "gpt35", "gpt4", "llama2_13bchat"] if dataset != "dolly" else ["human", "llama3_8bchat", "llama3_8bchat", "human"]
    cnn_inst = SUMMARIZE_PROMPT_TEMPLATE_CNN.split('\n\n')[1]
    xsum_inst = SUMMARIZE_PROMPT_TEMPLATE_XSUM.split('\n\n')[1]
    if dataset == "xsum":
        inst = xsum_inst
        responses, articles, keys = load_data("xsum")
    elif dataset == "cnn":
        inst = cnn_inst
        responses, articles, keys = load_data("cnn")
    else:
        inst = new_instruction
        responses, articles, instructions, keys = load_data_dolly()

    fewshot_keys = sorted(keys, reverse=True)[:len(models)]
    fewshot_articles, fewshot_instructions, fewshot_summaries = [], [], []
    
    for i, key in enumerate(fewshot_keys):
        fewshot_articles.append(articles[key])
        fewshot_summaries.append(responses[models[i]][key])   
        if dataset == "dolly": fewshot_instructions.append(instructions[key])
        else: fewshot_instructions.append(inst)
    
    input_text = f"<|start_header_id|>system<|end_header_id|>\n{DATASET_SYSTEM_PROMPTS[dataset]}<|eot_id|>"
    for article, instruction, summary in zip(fewshot_articles, fewshot_instructions, fewshot_summaries):
        input_text += f"<|start_header_id|>user<|end_header_id|>\n\nArticle:\n{article}\n\n{instruction}"
        input_text += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{summary.strip()}<|eot_id|>"
    input_text += f"<|start_header_id|>user<|end_header_id|>\n\nArticle:\n{new_article}\n\n{inst}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    return input_text
    
#_, articles, instructions, keys = load_data_dolly()
#input_text=make_fewshot_summaries_prompt("dolly", articles[keys[0]], instructions[keys[0]])
#print(input_text)

def make_fewshot_pairwise_detection_prompt(model_name, new_article, new_summary1, new_summary2, new_inst, suffix, task):
    from initialize import TaskType
    detect_prompt = DETECTION_PROMPT_TEMPLATE_INST if task == TaskType.SelfVOther else DETECTION_PROMPT_TEMPLATE_HUMAN_VS_MACHINE_INST
    models = ["human", "claude", "gpt35", "gpt4"]
    xsum_responses, xsum_articles, xsum_keys = load_data("xsum")
    cnn_responses, cnn_articles, cnn_keys = load_data("cnn")
    dolly_responses, dolly_articles, dolly_instructions, dolly_keys = load_data_dolly()

    fewshot_cnn_keys = sorted(cnn_keys, reverse=True)[90:92]#[:len(models)//2]
    fewshot_xsum_keys = sorted(xsum_keys, reverse=True)[90:92]#[:len(models)//2]
    fewshot_dolly_keys = sorted(dolly_keys, reverse=True)[90:92]#[:len(models)//2]
    articles, summaries1, summaries2, instructions = [], [], [], []

    # few-shot logic: cnn self first, xsum self first, xsum self second, cnn self second, dolly self first, dolly self second
    articles.append(cnn_articles[fewshot_cnn_keys[0]])
    summaries1.append(cnn_responses[model_name][fewshot_cnn_keys[0]])    
    summaries2.append(cnn_responses[models[0]][fewshot_cnn_keys[0]])   
    instructions.append(SUMMARIZE_PROMPT_TEMPLATE_CNN.split('\n\n')[1]) 
    articles.append(dolly_articles[fewshot_dolly_keys[0]])
    summaries1.append(dolly_responses[model_name][fewshot_dolly_keys[0]])    
    summaries2.append(dolly_responses[models[0]][fewshot_dolly_keys[0]])   
    instructions.append(dolly_instructions[fewshot_dolly_keys[0]]) 
#    articles.append(xsum_articles[fewshot_xsum_keys[0]])
#    summaries1.append(xsum_responses[model_name][fewshot_xsum_keys[0]])    
#    summaries2.append(xsum_responses[models[1]][fewshot_xsum_keys[0]])    
#    instructions.append(SUMMARIZE_PROMPT_TEMPLATE_XSUM.split('\n\n')[1]) 
    
#    articles.append(xsum_articles[fewshot_xsum_keys[1]])
#    summaries1.append(xsum_responses[models[2]][fewshot_xsum_keys[1]])    
#    summaries2.append(xsum_responses[model_name][fewshot_xsum_keys[1]])    
#    instructions.append(SUMMARIZE_PROMPT_TEMPLATE_XSUM.split('\n\n')[1]) 
    articles.append(dolly_articles[fewshot_dolly_keys[1]])
    summaries1.append(dolly_responses['human'][fewshot_dolly_keys[1]])    
    summaries2.append(dolly_responses[model_name][fewshot_dolly_keys[1]])    
    instructions.append(dolly_instructions[fewshot_dolly_keys[1]])
    articles.append(cnn_articles[fewshot_cnn_keys[1]])
    summaries1.append(cnn_responses[models[3]][fewshot_cnn_keys[1]])    
    summaries2.append(cnn_responses[model_name][fewshot_cnn_keys[1]])    
    instructions.append(SUMMARIZE_PROMPT_TEMPLATE_CNN.split('\n\n')[1]) 


    input_text = f"<|start_header_id|>system<|end_header_id|>\n{DETECTION_SYSTEM_PROMPT2}<|eot_id|>"
    for i, (article, summary1, summary2, inst) in enumerate(zip(articles, summaries1, summaries2, instructions)):
        if i == 0: input_text += f"<|start_header_id|>user<|end_header_id|>\n\n{detect_prompt.format(article=article, summary1=summary1.strip(), summary2=summary2.strip(), inst=inst)}"
        else: input_text += f"\n===========================\n\nArticle:\n{article}\n\nSummary1:\n{summary1.strip()}\n\nSummary2:\n{summary2.strip()}\n\n"
        input_text += f"{suffix}{(i//2)+1}"#f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>{suffix}{(i//2)+1}<|eot_id|>"
    input_text += f"<|start_header_id|>user<|end_header_id|>\n\n{detect_prompt.format(article=new_article, summary1=new_summary1, summary2=new_summary2, inst=new_inst)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    return input_text
    
#model_name = "llama3_8bchat"
#responses, articles, keys = load_data("cnn")
#responses, articles, instructions, keys = load_data_dolly()
#key = keys[2]
#self_response = responses[model_name][key]
#other_response = responses["human"][key]
#input_text=make_fewshot_pairwise_detection_prompt(model_name, articles[key], self_response, other_response, instructions[key], suffix="My answer is ")
#print(input_text)

def make_fewshot_individual_detection_prompt(model_name, new_article, new_summary, new_inst, suffix):
    resps = ["Yes", "Yes", "No", "No", "No", "Yes"]
    models = ["claude", "gpt35", "human"]

    xsum_responses, xsum_articles, xsum_keys = load_data("xsum")
    cnn_responses, cnn_articles, cnn_keys = load_data("cnn")
    dolly_responses, dolly_articles, dolly_instructions, dolly_keys = load_data_dolly()

    fewshot_cnn_keys = sorted(cnn_keys, reverse=True)[100:102]#[:len(models)]
    fewshot_xsum_keys = sorted(xsum_keys, reverse=True)[100:102]#[:len(models)]
    fewshot_dolly_keys = sorted(dolly_keys, reverse=True)[100:102]#[:len(models)]
    articles, summaries, insts = [], [], []

    # few-shot logic: cnn self-written, xsum self-written, xsum other-written, cnn other-written, dolly other-written, dolly self-written
    articles.append(cnn_articles[fewshot_cnn_keys[0]])    
    summaries.append(cnn_responses[model_name][fewshot_cnn_keys[0]]) 
    insts.append(SUMMARIZE_PROMPT_TEMPLATE_CNN.split('\n\n')[1])
    articles.append(dolly_articles[fewshot_dolly_keys[0]])    
    summaries.append(dolly_responses['human'][fewshot_dolly_keys[0]])    
    insts.append(dolly_instructions[fewshot_dolly_keys[0]])
#    articles.append(xsum_articles[fewshot_xsum_keys[0]])    
#    summaries.append(xsum_responses[model_name][fewshot_xsum_keys[0]])    
#    insts.append(SUMMARIZE_PROMPT_TEMPLATE_XSUM.split('\n\n')[1])
    
#    articles.append(xsum_articles[fewshot_xsum_keys[1]])    
#    summaries.append(xsum_responses[models[0]][fewshot_xsum_keys[1]])    
#    insts.append(SUMMARIZE_PROMPT_TEMPLATE_XSUM.split('\n\n')[1])
    articles.append(dolly_articles[fewshot_dolly_keys[1]])    
    summaries.append(dolly_responses[model_name][fewshot_dolly_keys[1]])    
    insts.append(dolly_instructions[fewshot_dolly_keys[1]])
    articles.append(cnn_articles[fewshot_cnn_keys[1]])    
    summaries.append(cnn_responses[models[1]][fewshot_cnn_keys[1]])    
    insts.append(SUMMARIZE_PROMPT_TEMPLATE_CNN.split('\n\n')[1])

    input_text = f"<|start_header_id|>system<|end_header_id|>\n{RECOGNITION_SYSTEM_PROMPT2}<|eot_id|>"
    for i, (article, summary, inst) in enumerate(zip(articles, summaries, insts)):
        input_text += f"<|start_header_id|>user<|end_header_id|>\n\n{RECOGNITION_PROMPT_TEMPLATE_ALT.format(article=article, summary=summary.strip(), inst=inst)}"
        input_text += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>{suffix}{resps[i]}<|eot_id|>"
    input_text += f"<|start_header_id|>user<|end_header_id|>\n\n{RECOGNITION_PROMPT_TEMPLATE_ALT.format(article=new_article, summary=new_summary, inst=new_inst)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    return input_text
    
#model_name = "llama3_8bchat"
#responses, articles, keys = load_data("cnn")
#responses, articles, instructions, keys = load_data_dolly()
#key = keys[2]
#article = articles[key]
#summary = responses["human"][key]
#input_text=make_fewshot_individual_detection_prompt(model_name, article, summary, instructions[key], "My answer is ")
#print(input_text)