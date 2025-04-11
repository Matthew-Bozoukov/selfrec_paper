Code and data for [INSPECTION AND CONTROL OF SELF-GENERATED-TEXT
RECOGNITION ABILITY IN LLAMA3-8B-INSTRUCT](https://arxiv.org/pdf/2410.02064)

"articles" directory contains dataset extracts (file_format = dict{id: text})

"summaries" directory contains model/human-written summaries/responses (file_pattern = {dataset}train{author}_responses.json; file_format = dict{id: text})

"*results directories contain model binary judgments from paired or individual presentation (file_pattern = {dataset}{model_name}_[pairwise/individual]_untuned.json; file_format = list[dict{"key": id, "model": model, [task]"_logprob": {token: logprob}, [task]: token, [author]"_summary_perplexity"})
