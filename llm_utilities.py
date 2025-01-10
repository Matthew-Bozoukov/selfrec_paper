### Functions to call models to generate log probs/perplexity

import torch
import torch.nn.functional as F

def generate_logprobs(model, tokenizer, input_text, tokens):
    """
    Runs the model on the input text and returns the log probabilities of the specified tokens.
    """
    # Prepare the input
    #print(f"Input text: {input_text}")
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(model.device)
    #print(f"Input ids (len {len(input_ids[0])}): {input_ids}")
    # Perform a forward pass
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract logits
    logits = outputs.logits

    # Select the logits for the first token position after the input
    first_position_logits = logits[0, len(input_ids[0]) - 1, :]

    # Apply softmax to get probabilities
    probs = F.softmax(first_position_logits, dim=-1)

    res = {}
    for token in tokens:
        res[token] = probs[tokenizer.encode(token, add_special_tokens=False)[-1]].item()

    return res

def generate_logprobs_batch(model, tokenizer, input_texts, tokens):
    """
    Runs the model on the input text and returns the log probabilities of the specified tokens, in batches.
    """
    # Prepare the inputs
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Perform a forward pass
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Extract logits
    logits = outputs.logits

    # Select the logits for the last token position of each input
    last_token_logits = logits[torch.arange(logits.shape[0]), attention_mask.sum(dim=1) - 1, :]

    # Apply softmax to get probabilities
    probs = F.softmax(last_token_logits, dim=-1)

    # Compute probabilities for the specified tokens
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_probs = probs[:, token_ids]

    return token_probs.cpu().numpy()


## Compare perplexities
##adapted from https://github.com/timoschick/self-debiasing/blob/main/perplexity.py
from torch.nn import CrossEntropyLoss
def compute_loss(model, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        outputs = model(input_ids, labels=labels)
        lm_logits = outputs[1]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
    
def compute_ppl(model, tokenizer, text, batch_size=4, max_length=1024, stride=512):
    encodings = tokenizer(text, return_tensors="pt")
    
    batch_len = len(encodings.input_ids[0]) // batch_size
    
    # Trim the tensor to fit complete sequences only
    input_tensor = encodings.input_ids[0][:batch_len * batch_size]
    
    # Reshape into batches
    batched_inputs = input_tensor.view(batch_size, -1)
    
    lls = []
    ppl = None
    for i in range(0, batched_inputs.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, batched_inputs.size(1))
        trg_len = end_loc - i  # stride, but may be different from stride on last loop
    
        input_ids = batched_inputs[:, begin_loc:end_loc].to(model.device)
    
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
    
        with torch.no_grad():
            loss = compute_loss(model, input_ids, labels=target_ids)
    
            log_likelihood = loss * trg_len
    
        lls.append(log_likelihood)
        ppl = torch.exp(torch.stack(lls).sum() / end_loc)
        #print(f"Perplexity after {i} tokens: {ppl}")
    return ppl.item()