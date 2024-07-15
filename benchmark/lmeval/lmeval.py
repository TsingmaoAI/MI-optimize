import os
import torch
import json
import datetime
import yaml
import torch.nn.functional as F

from benchmark.lmeval import utils
from tqdm import tqdm

def get_loglikelihood(model, tokenizer, requests, max_length=2048):
    args_list = []
    for req in requests:
        args_list.append(req.args)
        
    new_reqs = []
    for context, continuation in args_list:
        if context == "":
            context_enc = [2]
        else:
            context_enc = tokenizer.encode(context)

        continuation_enc = tokenizer.encode(continuation)
        new_reqs.append(((context, continuation), context_enc, continuation_enc))
    
    res = []

    ### loglikelihood need ###
    def _collate_loglikelihood(x):
        toks = x[1] + x[2]
        return -len(toks), tuple(toks)

    re_ord = utils.Reorderer(new_reqs, _collate_loglikelihood)
    for chunk in utils.chunks(tqdm(re_ord.get_reordered(), disable=False), 1):
        inps = []
        cont_toks_list = []
        inplens = []
        padding_length = None

        for _, context_enc, continuation_enc in chunk:
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= max_length

            inp = torch.tensor((context_enc + continuation_enc)[-(max_length + 1):][:-1], dtype=torch.long,).to(model.device)
            (inplen,) = inp.shape
            cont = continuation_enc
            padding_length = padding_length if padding_length is not None else inplen
            inp = torch.cat([inp, torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device),], dim=0,)
            inps.append(inp.unsqueeze(0))
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)


        multi_logits = F.log_softmax(model(batched_inps)["logits"], dim=-1).cpu() 

        for (_, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, cont_toks_list):
            contlen = len(cont_toks)
            logits = logits[inplen - contlen: inplen].unsqueeze(0)
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
            max_equal = (greedy_tokens == cont_toks).all()
            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
            answer = (float(logits.sum()), bool(max_equal))
            res.append(answer)
    
    return re_ord.get_original(res)


### loglikelihood_rolling need ###
def make_disjoint_window(pair):
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b

def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    first_seq_len = min(max_seq_len, len(token_list))
    yield ([prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len])
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len

def loglikelihood_rolling(model, tokenizer, requests, max_length=2048):
        loglikelihoods = []
        for string in tqdm(requests):
            rolling_token_windows = list(map(make_disjoint_window, get_rolling_token_windows(tokenizer.encode(string.args[0]), 2, max_length, 1)))
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)
            ## _loglikelihood_tokens
            res = []
            re_ord = utils.Reorderer(rolling_token_windows, _collate_loglikelihood)
            for chunk in utils.chunks(tqdm(re_ord.get_reordered(), disable=False), 1):
                inps = []
                cont_toks_list = []
                inplens = []
                padding_length = None

                for _, context_enc, continuation_enc in chunk:
                    assert len(context_enc) > 0
                    assert len(continuation_enc) > 0
                    assert len(continuation_enc) <= max_length

                    inp = torch.tensor((context_enc + continuation_enc)[-(max_length + 1):][:-1], dtype=torch.long,).to(model.device)
                    (inplen,) = inp.shape
                    cont = continuation_enc
                    padding_length = padding_length if padding_length is not None else inplen
                    inp = torch.cat([inp, torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device),], dim=0,)
                    inps.append(inp.unsqueeze(0))
                    cont_toks_list.append(cont)
                    inplens.append(inplen)

                batched_inps = torch.cat(inps, dim=0)

                # Most Valuable Code(@Pshenhao @time 2024/05/10)
                multi_logits = F.log_softmax(model(batched_inps)["logits"], dim=-1).cpu() 

                for (_, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, cont_toks_list):
                    contlen = len(cont_toks)
                    logits = logits[inplen - contlen: inplen].unsqueeze(0)
                    greedy_tokens = logits.argmax(dim=-1)
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                    max_equal = (greedy_tokens == cont_toks).all()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)
                    answer = (float(logits.sum()), bool(max_equal))
                    res.append(answer)
            
            string_nll = re_ord.get_original(res)
            string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods


def greedy_until(model, tokenizer, requests, max_length=2048):
    results = []

    ### greedy_until need ###
    def _collate_greed(x, tokenizer=tokenizer):
        x_str = x.args[0]
        toks = tokenizer.encode(x_str)
        return len(toks), x_str

    reorder = utils.Reorderer(requests, _collate_greed)
    for chunk in utils.chunks(tqdm(reorder.get_reordered(), disable=False), 1):
        context = [c.args[0] for c in chunk]
        until = ['</s>']
        token_context = tokenizer(context[0], return_tensors='pt')

        input_ids = token_context["input_ids"][:, 256 - max_length :]
        attention_mask = token_context["attention_mask"][:, 256 - max_length :]
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=256, do_sample=False)
        # responses = tokenizer.encode(responses.tolist())
        responses = tokenizer.decode(output[0], skip_special_tokens=True)
        # print('responses', responses)
        for response in responses:
            for term in until:
                response = response.split(term)[0]
            results.append(response)
    return reorder.get_original(results)



def keep_result(results, model_name, algorithm_name, w_bit, calibrate_name):
    """
    @results: Test results for each subject
    """
    formatted_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./log/{model_name}_{algorithm_name}_{calibrate_name}_{w_bit}_{formatted_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    acc_path = os.path.join(output_dir, "acc.json")
    
    # write result to file
    with open(acc_path, 'w') as f:
        json.dump(results, f, indent=4)
