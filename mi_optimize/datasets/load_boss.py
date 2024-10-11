import os
import random
import json
import csv 
import yaml

from itertools import islice

from click import prompt
from collections import Counter

def get_boss(task_name, dataset_name, split, shuffle=False, seed=42):
    SA_label_mapping = {"0":"negative", "1":"positive", "2":"neutral"}
    NLI_label_mapping = {"0":"entailment", "1":"neutral","2":"contradiction"}
    TD_label_mapping = {"0":"benign","1":"toxic"}
    data_dir = f"mi_optimize/datasets/BOSS/{task_name}/{dataset_name}"
    if task_name == "QuestionAnswering":
        examples = []
        with open(os.path.join(data_dir, f"{split}.json"),'r') as f:
            lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            processed_data = {'id': json_data['id'], 'title': json_data['title'], 'context': json_data['context'], 'question': json_data['question'], 'answers': json_data['answers']['text']}
            examples.append(processed_data) 

    elif task_name == "SentimentAnalysis":
        examples = []
        with open(os.path.join(data_dir, f"{split}.tsv"),'r',newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row and row['Label']:
                    context = row["Text"]
                    answer = row['Label']
                    processed_data = {'context': context, 'answer': SA_label_mapping[answer]}
                    examples.append(processed_data)

    elif task_name == "NaturalLanguageInference":
        examples = []
        with open(os.path.join(data_dir, f"{split}.tsv"),'r',newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row and row['Label']:
                    premise = row['Premise']
                    hypothesis = row['Hypothesis']
                    label = row['Label']
                    processed_data = {'premise': premise, 'hypothesis': hypothesis, 'label': NLI_label_mapping[label]}
                    examples.append(processed_data)
    elif task_name == "ToxicDetection":
        examples = []
        with open(os.path.join(data_dir, f"{split}.tsv"),'r',newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row and row['Label']:
                    context = row["Text"]
                    answer = row['Label']
                    processed_data = {'context': context, 'answer': TD_label_mapping[answer]}
                    examples.append(processed_data)
    elif task_name == "NameEntityRecognition":
        pass
                    
    if shuffle:
        random.seed(seed)
        datasets = list(datasets)
        random.shuffle(datasets)

    return examples


def get_str(task_name, dataset_name, nsamples=208, split='train', shuffle=False, seed=42):

    question_list = []
    if task_name == "QuestionAnswering": 
        examples = get_boss(task_name,dataset_name,split=split,shuffle=shuffle,seed=seed)
        question_num = nsamples if nsamples < len(examples) else len(examples)
        for example in islice(examples, question_num):
            title_str = example['title']
            context_str = example['context']
            question_str = example['question']
            answers_str = example['answers']
            question_str=(f"Question:{question_str} Context:{context_str} Answer:{answers_str}")
            question_list.append(question_str)

    elif task_name == "SentimentAnalysis":
        examples = get_boss(task_name,dataset_name,split=split,shuffle=shuffle,seed=seed)
        question_num = nsamples if nsamples < len(examples) else len(examples)
        for example in islice(examples,question_num):
            context = example["context"]
            answer = example["answer"]
            question_str = (f"Text:{context} Label:{answer}")
            question_list.append(question_str)
    
    elif task_name == "NaturalLanguageInference":
        examples = get_boss(task_name,dataset_name,split=split,shuffle=shuffle,seed=seed)
        question_num = nsamples if nsamples < len(examples) else len(examples)
        for example in islice(examples,question_num):
            premise = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']
            question_str = (f"Premise:{premise} Hypothesis:{hypothesis} Label:{label}")
            question_list.append(question_str)
    
    elif task_name == "ToxicDetection":
        examples = get_boss(task_name,dataset_name,split=split,shuffle=shuffle,seed=seed)
        question_num = nsamples if nsamples < len(examples) else len(examples)
        for example in islice(examples,question_num):
            context = example["context"]
            answer = example["answer"]
            question_str = (f"Text:{context} Label:{answer}")
            question_list.append(question_str)
    
    elif task_name == "NameEntityRecognition":
        pass

    return question_list



def get_calibrate_boss(tokenizer, task_name, dataset_name, calibrate_nums=128, split='train', shuffle=False, seed=42, calibrate_seqlen=2048, **kwargs):
    calibrate_data = get_str(task_name, dataset_name, calibrate_nums,split,shuffle,seed)
    inputs_ids = []
    for data in calibrate_data:
        input_ids = tokenizer.encode(data, return_tensors='pt')
        inputs_ids.append(input_ids[:calibrate_seqlen])
    return inputs_ids

def get_fewshot_boss(task_name, dataset_name, fewshot_num, split='test', answer=True, max_length=1024, shuffle=False, seed=42):

    if task_name == "QuestionAnswering": 
        prompt = f"### Instruction ###\n\
Solve the extractive question answering task. Refering to the passage below and extract answer for the question. The answer should be the shortest phrase as it can be.\n\
### Format ###\n\
Passage: {{Passage}} // Question: {{Question}} // Answer: {{Answer}}.\n\
### Example ###\n"
        if dataset_name == 'squad':
            example = "Passage: Everton F.C. is a limited company with the board of directors holding a majority of the shares. The club\'s most recent accounts, from May 2014, show a net total debt of £28.1 million, with a turnover of £120.5 million and a profit of £28.2 million. The club\'s overdraft with Barclays Bank is secured against the Premier League\'s \"Basic Award Fund\", a guaranteed sum given to clubs for competing in the Premier League. Everton agreed a long-term loan of £30 million with Bear Stearns and Prudential plc in 2002 over the duration of 25 years; a consolidation of debts at the time as well as a source of capital for new player acquisitions. Goodison Park is secured as collateral. // Question: How long does Everton FC have to pay back £30 million they borrowed from Bear Stearns and Prudential? // Answer: 25 years.\n"
        
        if dataset_name == 'advqa':
            example = "Passage: The northeastern Puntland region has around six private radio stations, including Radio Garowe, Radio Daljir, Radio Codka-Nabbada and Radio Codka-Mudug. Radio Gaalkacyo, formerly known as Radio Free Somalia, operates from Galkayo in the north-central Mudug province. Additionally, the Somaliland region in the northwest has one government-operated radio station. // Question: Hassan Moalim is chairman of the __ Party. // Answer: Daljir.\n"
            
        if dataset_name == 'newsqa':
            example = "Passage: TOKYO, Japan (CNN) -- Typhoon Melor roared into central Japan on Thursday, leaving two people dead and lashing the region with heavy rain and gusty winds.\n\n\n\nUtility poles lie buckled in the wake of Typhoon Melor.\n\n\n\nThe storm stayed west of Tokyo, but still caused enough trouble to shut down trains for a time and snarl commuter traffic. Numerous flights were canceled and delayed at the city's two major airports.\n\n\n\nIn western and northern  Japan, Melor tore roofs off homes, downed power lines and flooded roads.\n\n\n\nThe storm contributed to the deaths of a 54-year-old newspaper delivery man in Wakayama, who ran into a fallen tree, and a 69-year-old man from Saitama, who was crushed by a tree.\n\n\n\nBy late Thursday, Melor had weakened to a tropical storm and was heading out to sea.\n\n\n\n-- CNN's Kyung Lah contributed to this report. // Question: What did the storm avoid? // Answer: Tokyo.\n"
            
        if dataset_name == 'searchqa': # We designed ourselves.
            example = "Passage: [DOC] [TLE] Jeopary Questions page 1085 - \"N\" GAME - TriviaBistro.com [PAR] AROUND THE WORLD: Europe's Jostedal glacier has shrunk so much, you can \nsee farms ... WE ARE YOUNG: He was a teenage cabin boy in the 1790s but \nsoon realized that \"bird artist\" had more prestige than \"cabin boy\" ... Easton sang, \n\"My baby takes\" this, \"he works from 9 till 5 & then he takes another home again\". [DOC] [TLE] Free Flashcards about ART - StudyStack [PAR] HE WAS A TEENAGE CABIN BOY IN THE 1790S, BUT SOON REALIZED THAT \"\nBIRD ARTIST\" HAD MORE PRESTIGE THAN \"CABIN BOY\", AUDUBON. // Question: He was a teenage cabin boy in the 1790s but soon realized that \"bird artist\" had more prestige than \"cabin boy\" // Answers: Audubon.\n"
            
        prompt = prompt + example + "### Input ###\n"


    elif task_name == "SentimentAnalysis":
        prompt = f"### Instruction ###\n\
Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\n\
### Format ###\n\
Text: {{Text}} // Prediction: {{Prediction}}\n\
### Example ###\n"

        if dataset_name == "dynasent":
            example = "Text: I\'ve been here for dine-in and take out and bad experiences both times. // Prediction: negative\n\
Text: I remember going to this place back in 2014 and loved the food, the price and the portions. // Prediction: positive\n\
Text: There is a good chance that I will do the repair since I already used your services earlier. // Prediction: neutral\n"

        if dataset_name == "sst5":
            example = "Text: It \'s a stale , overused cocktail using the same olives since 1962 as garnish . // Prediction: negative\n\
Text: It \'s a great American adventure and a wonderful film to bring to IMAX . // Prediction: positive\n\
Text: What makes the movie work -- to an admittedly limited extent -- is the commitment of two genuinely engaging performers . // Prediction: neutral\n"

        if dataset_name == "semeval":
            example = "Text: David Beckham as James Bond may just be the worst idea I\'ve ever heard in my life // Prediction: negative\n\
Text: #LinuxCon attendees, join IBM in the Cedar Room in 15 minutes for a \"Linux Without Limits\" meetup (includes lunch): http://t.co/q5XYwe8Bkd // Prediction: positive\n\
Text: SpeedeNews: Rumor: Android Wear will soon work with iOS - Android Wear may be very close to working with iOS. That... http://t.co/pq0g0Z3eOy // Prediction: neutral\n"

        if dataset_name == "amazon":
            example = "Text: only lasted 7 months and had to replace again // Prediction: negative\n\
Text: Appears to be excellent quality. Haven\'t used yet. // Prediction: positive\n\
Text: There aren\'t many veggie chips in the bag but they are very tasty. // Prediction: neutral\n"

        prompt = prompt + example + "### Input ###\n"


    elif task_name == "NaturalLanguageInference":
        prompt = f"### Instruction ###\n\
Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\n\
### Format ###\n\
Premise: {{Premise}} // Hypothesis: {{Hypothesis}} // Prediction: {{Prediction}}\n\
### Example ###\n"
        
        if dataset_name == "anli":
            example = "Premise: At the time, neither \"USA Today\" nor \"The Observer\" was able to verify which of the sixty-plus candidates it was. // Hypothesis: It says that they could not verify it // Prediction: entailment\n\
Premise: Cyprus International Football Tournaments is annual winter association football friendly competition for national teams that take place in Cyprus. Traditionally held in February as a friendly tournament since at least 1997. Played in two parallel groups in 2006. // Hypothesis: the agent is aged 37 // Prediction: neutral\n\
Premise: Ugly Cat<br>Tom noticed a stray cat. It was really ugly. Tom went to pet it anyway. The cat was super friendly and nice. Tom decided to keep it as a pet. // Hypothesis: Tom contains a x // Prediction: contradiction\n"

        if dataset_name == "contractnli": # We use MNLI's prompts
            example = "Premise: As the roc swept over, the people stopped their frenzied pursuit of sensation and ran for weapons. // Hypothesis: As the roc swept over, people ran to grab weapons. // Prediction: entailment\n\
Premise: What they do know, however, is that men's chins have been getting larger over the last 200 generations. // Hypothesis: They were trying to figure out why the chins were getting bigger. // Prediction: neutral\n\
Premise: With the passing of each year, the Space Needle looks more and more like a prop from a bad science-fiction movie. // Hypothesis: The Space Needle has been updated and modernized. // Prediction: contradiction\n"

        if dataset_name == "mnli":
            example = "Premise: As the roc swept over, the people stopped their frenzied pursuit of sensation and ran for weapons. // Hypothesis: As the roc swept over, people ran to grab weapons. // Prediction: entailment\n\
Premise: What they do know, however, is that men's chins have been getting larger over the last 200 generations. // Hypothesis: They were trying to figure out why the chins were getting bigger. // Prediction: neutral\n\
Premise: With the passing of each year, the Space Needle looks more and more like a prop from a bad science-fiction movie. // Hypothesis: The Space Needle has been updated and modernized. // Prediction: contradiction\n"

        if dataset_name == "wanli":
            example = "Premise: By the time the boys reached the fire, the wind had already taken the roof off the house. // Hypothesis: The wind had destroyed the house. // Prediction: entailment\n\
Premise: \"I don't know what I'm doing,\" she said. \"I'm just trying to be a good person.\" // Hypothesis: She is not a good person. // Prediction: neutral\n\
Premise: The two brothers, then, must have been in their early twenties. // Hypothesis: The two brothers were younger than 20. // Prediction: contradiction\n"

        prompt = prompt + example + "### Input ###\n"


    elif task_name == "ToxicDetection":
        prompt = f"### Instruction ###\n\
Solve the toxic detection task. Options for toxicity: benign, toxic.\n\
### Format ###\n\
Text: {{Text}} // Prediction: {{Prediction}}\n\
### Example ###\n"

        if dataset_name == "advcivil": # We designed ourselves.
            example = "Text: srue... take the cops wirh them and then we an coke to your house and take everything we want.. Be careful what you say you want.. It may not be what you really want. , Die // Prediction: benign\n\
Text: \"Yo Goldberg: ""I have no ide wat's going on"" truer words were never spoken by you. Well, meafthead, I'll tel you what's going on. The POTUS is kicking sas and taking names, and so, even you will eventually realize that's whats going on is good for the goose and the gander. 's ensemble\" // Prediction: toxic\n"

        if dataset_name == "civilcomments":
            example = "Text: Headline is accurate. Not sure how your arrived at the 7 year number. He's been an Alaskan since 1986, save for the time he was in college. He's worked in state government for nearly 20 years but the last two working for Sen. Sullivan required him to move to the D.C. area. // Prediction: benign\n\
Text: Never heard of this person and I'm sure the majority haven't either. Once her little following of \"woke\" teenagers grow up and realize social justice is a stupid fad she'll be waiting in the bread line along with her communist friends. // Prediction: toxic\n"

        if dataset_name == "implicithate": # We designed ourselves.
            example = "Text: : all because one man has the audacity to think it's ok to be white and ok to not hate you // Prediction: benign\n\
Text: white will dominate ! sieg heil hitler // Prediction: toxic\n"

        if dataset_name == "toxigen":
            example = "Text: the british knew that large american cities such as virginia were a prime target during the revolutionary war // Prediction: benign\n\
Text: If somebody has a physical disability they are unpredictable and should not be allowed near sharp objects // Prediction: toxic\n"

        prompt = prompt + example + "### Input ###\n"


    return prompt

def get_zeroshot_boss(task_name, dataset_name):
    if task_name == "QuestionAnswering":
        prompt = f"### Instruction ###\n\
Solve the extractive question answering task. Refering to the passage below and extract answer for the question. The answer should be the shortest phrase as it can be.\n\
### Format ###\n\
Passage: {{Passage}} // Question: {{Question}} // Answer: {{Answer}}.\n\
### Input ###\n"
    
    elif task_name == "SentimentAnalysis":
        prompt = f"### Instruction ###\n\
Solve the sentiment analysis task. Options for sentiment: negative, positive, neutral.\n\
### Format ###\n\
Text: {{Text}} // Prediction: {{Prediction}}\n\
### Input ###\n"

    elif task_name == "NaturalLanguageInference":
        prompt = f"### Instruction ###\n\
Solve the NLI task. Options for entailment relationship: entailment, neutral, contradiction.\n\
### Format ###\n\
Premise: {{Premise}} // Hypothesis: {{Hypothesis}} // Prediction: {{Prediction}}\n\
### Input ###\n"

    elif task_name == "ToxicDetection":
        prompt = f"### Instruction ###\n\
Solve the toxic detection task. Options for toxicity: benign, toxic.\n\
### Format ###\n\
Text: {{Text}} // Prediction: {{Prediction}}\n\
### Input ###\n"

    elif task_name == "NameEntityRecognition":
        pass

    return prompt

def get_testdata_boss(task_name, dataset_name, split, model_name='llama', answer=True, shuffle=False, seed=42):
    question_list = []
    answer_list = []
    if task_name == "QuestionAnswering": 
        qa_data = get_boss(task_name, dataset_name,split=split,shuffle=shuffle,seed=seed)
        for data in qa_data:
            question = f"Passage: {data['context']} // Question: {data['question']} // Answer:"
            question_list.append(question)
            answer_list.append(data['answers'])

    elif task_name == "SentimentAnalysis":
        examples = get_boss(task_name,dataset_name,split=split,shuffle=shuffle,seed=seed)
        for example in examples:
            question = f"Text: {example['context']} // Prediction:"
            question_list.append(question)
            answer_list.append(example['answer'])

    elif task_name == "NaturalLanguageInference":
        examples = get_boss(task_name,dataset_name,split=split,shuffle=shuffle,seed=seed)
        for example in examples:
            question = f"Premise: {example['premise']} // Hypothesis: {example['hypothesis']} // prediction:"
            question_list.append(question)
            answer_list.append(example['label']) 
    
    elif task_name == "ToxicDetection":
        examples = get_boss(task_name,dataset_name,split=split,shuffle=shuffle,seed=seed)
        for example in examples:
            question = f"Text: {example['context']} // Prediction:"
            question_list.append(question)
            answer_list.append(example['answer'])

    elif task_name == "NameEntityRecognition":
        pass
    
    return question_list, answer_list

