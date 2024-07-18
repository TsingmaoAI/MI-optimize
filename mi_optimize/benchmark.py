import logging
import torch
import numpy as np
import collections
from tqdm import tqdm

from mi_optimize.datasets import get_wikitext2, get_ptb, get_c4
from mi_optimize.datasets.load_ceval import get_subjects_ceval, get_testdaset_ceval, get_fewshot_ceval, extract_cot_answer_ceval, classifi_results_ceval
from mi_optimize.datasets.load_cmmlu import get_subjects_cmmlu, get_testdata_cmmlu, get_fewshot_cmmlu, extract_cot_answer_cmmlu, classifi_results_cmmlu
from mi_optimize.datasets.load_boss import get_fewshot_boss, get_zeroshot_boss, get_testdata_boss
from benchmark.boss.metrics import compute_metric
from transformers import pipeline
logging.basicConfig(level=logging.INFO)

class Benchmark:
    def __init__(self):
        pass
    
    @torch.no_grad()
    def compute_ppl(self, model, tokenizer, loader):
        total_loss = 0
        total_count = 0
        for batch in tqdm(loader):
            batch = batch.clone()
            if batch.shape[1] <= 1: continue
            input_ids = batch.to(model.device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            if tokenizer.pad_token_id is not None:
                
                count = input_ids.ne(tokenizer.pad_token_id).ne(-100).sum().item()
            else:
                count = input_ids.ne(-100).sum().item()
            total_loss += loss.item() * count
            total_count += count
                
        return np.exp(total_loss / total_count)
    
    def eval_wiki2_ppl(self, model, tokenizer, nsamples='all', split='test'):
        logging.info("Evaluating Perplexity (PPL) on the wikitext2")
        dataloader = get_wikitext2(tokenizer, nsamples=nsamples, split=split)
        ppl = self.compute_ppl(model, tokenizer, dataloader)
        logging.info(f'wikitext2 PPL {ppl}')
        return ppl
    
    def eval_ptb_ppl(self, model, tokenizer, nsamples='all', split='test'):
        logging.info("Evaluating Perplexity (PPL) on the ptb")
        dataloader = get_ptb(tokenizer, nsamples=nsamples, split=split)
        ppl = self.compute_ppl(model, tokenizer, dataloader)
        logging.info(f'ptb PPL {ppl}')
        return ppl
    
    def eval_c4_ppl(self, model, tokenizer, nsamples='all', split='validation'):
        logging.info("Evaluating Perplexity (PPL) on the c4")
        dataloader = get_c4(tokenizer, nsamples=nsamples, split=split)
        ppl = self.compute_ppl(model, tokenizer, dataloader)
        logging.info(f'c4 PPL {ppl}')
        return ppl
    
    def eval_ppl(self, model, tokenizer, nsamples='all', test_datasets=['wikitext2', 'ptb']):
        logging.info("Evaluating Perplexity (PPL) on the wikitext2, c4, ptb dataset")
        results = {}
        if 'wikitext2' in test_datasets: 
            wiki2_ppl = self.eval_wiki2_ppl(model, tokenizer, nsamples=nsamples)
            results['wikitext_ppl'] = wiki2_ppl
        if 'ptb' in  test_datasets:
            ptb_ppl = self.eval_ptb_ppl(model, tokenizer, nsamples=nsamples)
            results['ptb_ppl'] = ptb_ppl
        if 'c4' in test_datasets:
            c4_ppl = self.eval_c4_ppl(model, tokenizer, nsamples=nsamples)
            results['c4_ppl'] = c4_ppl
        return results
    
    def eval_ceval(self, model, tokenizer, model_type='baichuan', subject='all', split='val',num_shot=0):
        results = {}
        subject_dict = get_subjects_ceval(subject)
        for subject in tqdm(subject_dict):
            question_list, answer_list = get_testdaset_ceval(subject=[subject], split=split)
            count = 0
            correct = 0
            for question, answer in zip(question_list, answer_list):
                if num_shot:
                    question_prompt = get_fewshot_ceval(subject, model_name=model_type, question=num_shot)
                else:
                    question_prompt = ""
                
                if model_type=='chatglm':
                    response, _ = model.chat(tokenizer, question, do_sample=False, history=question_prompt)
                    response = response.strip()
                    response_answer = extract_cot_answer_ceval(question, response)

                elif model_type=='baichuan' or model_type=='llama':
                    question = question_prompt + "\n\n" + question
                    inputs = tokenizer(question, return_tensors='pt')
                    input_ids = inputs['input_ids'].to(model.device)
                    attention_mask = inputs['attention_mask'].to(model.device)
                    output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, temperature=0.8, top_p=0.95, pad_token_id=tokenizer.eos_token_id)

                    scores = output.scores[0][0].to(torch.float32)
                    label_score = []
                    candidates = ["A", "B", "C", "D"]
                    for can in candidates:
                        can_id = tokenizer.encode(can)[-1]
                        label_score.append(scores[can_id].item())
                    response_answer = candidates[np.argmax(label_score)]

                else:
                    raise ValueError(f'not support {model_type}')
                
                count = count + 1
                correct += (answer == response_answer)

            ratio = correct / count
            res_str = f"correct: {correct} total: {count} ratio: {ratio}"
            results.update({f"{subject}": res_str})

            # print(res_str)
        
        all_results = classifi_results_ceval(results)
        return all_results
    
    def eval_cmmlu(self, model, tokenizer, model_type='baichuan', subject='all', split='test', num_shot=0):
        results = {}
        subject_dict = get_subjects_cmmlu(subject)
        for subject in tqdm(subject_dict):
            question_list, answer_list = get_testdata_cmmlu(subject=[subject], split=split)
            count = 0
            correct = 0
            for question, answer in zip(question_list, answer_list):
                if num_shot:
                    question_prompt = get_fewshot_cmmlu(subject, model_name=model_type, question=num_shot)
                else:
                    question_prompt = ""

                if model_type=='chatglm':
                    response, _ = model.chat(tokenizer, question, do_sample=False, history=question_prompt)
                    response = response.strip()
                    response_answer = extract_cot_answer_cmmlu(question, response)

                elif model_type=='baichuan' or model_type=='llama':
                    question = question_prompt + "\n\n" + question
                    inputs = tokenizer(question, return_tensors='pt')
                    input_ids = inputs['input_ids'].to(model.device)
                    attention_mask = inputs['attention_mask'].to(model.device)
                    output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, temperature=0.1, top_p=0.5, repetition_penalty=1.1, pad_token_id=tokenizer.eos_token_id)
                    
                    scores = output.scores[0][0].to(torch.float32)
                    label_score = []
                    candidates = ["A", "B", "C", "D"]
                    for can in candidates:
                        can_id = tokenizer.encode(can)[-1]
                        label_score.append(scores[can_id].item())
                    response_answer = candidates[np.argmax(label_score)]

                else:
                    raise ValueError(f'not support {model_type}')

                count = count + 1
                correct += (answer == response_answer)
                if subject not in results:
                    results[subject] = {}
                results[subject][str(count - 1)] = response_answer

            ratio = correct / count
            res_str = f"correct: {correct} total: {count} ratio: {ratio}"
            results.update({f"{subject}": res_str})

            # print(res_str)
        
        all_results = classifi_results_cmmlu(results)
        return all_results

    def eval_boss(self, model, tokenizer, test_dataset, split='test', ICL_split='test', num_shot=0):
        MAX_TOKENS = {
            "SentimentAnalysis": 2,
            "ToxicDetection": 1,
            "NaturalLanguageInference": 1,
            "NameEntityRecognition": 50,
            "QuestionAnswering": 5
        }

        logging.info("Evaluating the model on the boss benchmark")

        generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device='cuda')

        results = {}
        test_dataset = test_dataset.split("_")
        task_name = test_dataset[0]
        dataset_name = test_dataset[1]
        question_list, answer_list = get_testdata_boss(task_name, dataset_name, split=split)

        if num_shot:         
            question_prompt = get_fewshot_boss(task_name, dataset_name, num_shot, ICL_split)
        else:
            question_prompt = get_zeroshot_boss(task_name, dataset_name)

        prediction_list = []
        for question in tqdm(question_list):
            question = question_prompt + question
            output = generator(question, num_return_sequences=1, return_full_text=False, handle_long_generation="hole",
                               temperature=0, max_new_tokens=MAX_TOKENS[task_name], do_sample=False, pad_token_id=tokenizer.eos_token_id)
            output = output[0]["generated_text"].strip("\n").strip()

            prediction_list.append(output)

        results = compute_metric(task_name, prediction_list, answer_list)

        return results

    def eval_lmeval(self, model, tokenizer, eval_tasks, num_shot):
        from mi_optimize.datasets.load_lmeval import get_testdata
        from benchmark.lmeval.lmeval import get_loglikelihood, greedy_until, loglikelihood_rolling
        return_dict, tasks_dict = get_testdata(eval_tasks, num_shot)
        docs = return_dict['docs']  
        versions = return_dict['versions']
        requests = return_dict['requests']
        requests_origin = return_dict['requests_origin']
        results = collections.defaultdict(dict)
        process_res_queue = collections.defaultdict(list)
        for reqtype, reqs in requests.items():
            print("Running", reqtype, "requests")
            if reqtype == 'loglikelihood':
                resps = get_loglikelihood(model, tokenizer, reqs)
            elif reqtype == 'greedy_until':
                resps = greedy_until(model, tokenizer, reqs)
            elif reqtype == 'loglikelihood_rolling':
                resps = loglikelihood_rolling(model, tokenizer, reqs)
    
            resps = [x if req.index is None else x[req.index] for x, req in zip(resps, reqs)]

            for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
                process_res_queue[(task_name, doc_id)].append((i, resp))
            

        vals = collections.defaultdict(list)
        for (task_name, doc_id), requests in process_res_queue.items():
            requests.sort(key=lambda x: x[0])
            requests = [x[1] for x in requests]
            task = tasks_dict[task_name]
            doc = docs[(task_name, doc_id)]

            metrics = task.process_results(doc, requests)
            for metric, value in metrics.items():
                vals[(task_name, metric)].append(value)

        for (task_name, metric), items in vals.items():
            task = tasks_dict[task_name]
            real_metric = metric
            if metric.endswith("_decontaminate"):
                real_metric = metric.replace("_decontaminate", "")
            results[task_name][metric] = task.aggregation()[real_metric](items)

        results_json = {"results": dict(results), "versions": dict(versions)}
        print(results_json)
        return results_json

