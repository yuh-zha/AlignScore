from logging import error
from datasets import load_dataset
import transformers
from random import sample
import random
import torch
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import re


'''
data format
{text_a, text_b, label:None or 0_1, }
'''
DATASET_HUGGINGFACE = {
    'cnndm': ['cnn_dailymail', '3.0.0', 'train'],
    'mnli': ['multi_nli', 'default', 'train'],
    'squad': ['squad', 'plain_text', 'train'],
    'squad_v2': ['squad_v2', 'squad_v2', 'train'],
    'paws': ['paws', 'labeled_final', 'train'],
    'vitaminc': ['tals/vitaminc', 'v1.0', 'train'],
    'xsum': ['xsum', 'default', 'train'],
    'stsb': ['glue', 'stsb', 'train'],
    'sick': ['sick', 'default', 'train'],
    'race': ['race', 'all', 'train'],
    'race_val': ['race', 'all', 'validation'],
    'anli_r1': ['anli', 'plain_text', 'train_r1'],
    'anli_r2': ['anli', 'plain_text', 'train_r2'],
    'anli_r3': ['anli', 'plain_text', 'train_r3'],
    'snli': ['snli', 'plain_text', 'train'],
    'wikihow': ['wikihow', 'all', 'train'],
    'mrpc': ['glue', 'mrpc', 'train'],
    'msmarco': ['ms_marco', 'v2.1', 'train'],
    'mrpc_val': ['glue', 'mrpc', 'validation'],
    'paws_val': ['paws', 'labeled_final', 'validation'],
    'paws_unlabeled': ['paws', 'unlabeled_final', 'train'],
    'qqp': ['glue', 'qqp', 'train'],
    'qqp_val': ['glue', 'qqp', 'validation'],
    'squad_v2_new': ['squad_v2', 'squad_v2', 'train'],
    'adversarial_qa': ['adversarial_qa', 'adversarialQA', 'train'],
    'drop': ['drop', 'train'],
    'duorc_self': ['duorc', 'SelfRC', 'train'],
    'duorc_paraphrase': ['duorc', 'ParaphraseRC', 'train'],
    'quoref': ['quoref', 'train'],
    'hotpot_qa_distractor': ['hotpot_qa', 'distractor', 'train'],
    'hotpot_qa_fullwiki': ['hotpot_qa', 'fullwiki', 'train'],
    'ropes': ['ropes', 'train'],
    'boolq': ['boolq', 'train'],
    'eraser_multi_rc': ['eraser_multi_rc', 'train'],
    'quail': ['quail', 'train'],
    'sciq': ['sciq', 'train'],
    'strategy_qa': ['metaeval/strategy-qa', 'train'],
    'gap': ['gap', 'train'],
}

DATASET_CONFIG = {
    'cnndm': {'task': 'summarization', 'text_a': 'article', 'text_b': 'highlights', 'label': None, 'huggingface': True},
    'mnli': {'task': 'nli', 'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'huggingface': True},
    'nli_fever': {'task': 'fact_checking', 'text_a': 'context', 'text_b': 'query', 'label': 'label','huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json':True, 'data_path':'data/nli_fever/train_fitems.jsonl' },
    'doc_nli': {'task': 'bin_nli', 'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label','huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json':True, 'data_path':'data/DocNLI_dataset/train.json' },
    'squad': {'task': 'extractive_qa', 'text_a': 'context', 'text_b': ['question', 'answers'], 'label': None, 'huggingface': True},
    'squad_v2': {'task': 'qa', 'text_a': 'context', 'text_b': ['question', 'answers'], 'label': None, 'huggingface': True},
    'paws': {'task': 'paraphrase', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'huggingface': True},
    'vitaminc': {'task': 'fact_checking', 'text_a': 'evidence', 'text_b': 'claim', 'label': 'label', 'huggingface': True},
    'xsum': {'task': 'summarization', 'text_a': 'document', 'text_b': 'summary', 'label': None, 'huggingface': True, 'cliff_path': 'data/model_generated_data/cliff_summ/xsum_train.jsonl'},
    'stsb': {'task': 'sts', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'huggingface': True},
    'sick': {'task': 'sts', 'text_a': 'sentence_A', 'text_b': 'sentence_B', 'label': 'relatedness_score', 'huggingface': True},
    'race': {'task': 'qa', 'text_a': 'article', 'text_b': ['question', 'options'], 'label': 'answer', 'huggingface': True},
    'race_val': {'task': 'qa', 'text_a': 'article', 'text_b': ['question', 'options'], 'label': 'answer', 'huggingface': True},
    'anli_r1': {'task': 'nli', 'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'huggingface': True},
    'anli_r2': {'task': 'nli', 'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'huggingface': True},
    'anli_r3': {'task': 'nli', 'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'huggingface': True},
    'snli': {'task': 'nli', 'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'huggingface': True},
    'wikihow': {'task': 'summarization', 'text_a': 'text', 'text_b': 'headline', 'label': None, 'huggingface': False, 'using_hf_api': True, 'data_dir': 'data/wikihow_raw'},
    'mrpc': {'task': 'paraphrase', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label','huggingface': True},
    'mrpc_val': {'task': 'paraphrase', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label','huggingface': True},
    'paws_val': {'task': 'paraphrase', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'huggingface': True},
    'paws_unlabeled': {'task': 'paraphrase', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'huggingface': True},
    'msmarco': {'task': 'ir', 'text_a': 'passages', 'text_b': ['query', 'answers'], 'label': None,'huggingface': True},
    'paws_qqp': {'task': 'paraphrase', 'text_a': 'sentence1', 'text_b': 'sentence2', 'label': None,'huggingface': False, 'using_hf_api': False, 'using_pandas': True, 'data_path':'paws_qqp/output/train.tsv' },
    'wiki103': {'task': 'paraphrase', 'text_a': 'original_sent', 'text_b': 'paraphrase', 'label': None,'huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json': True, 'data_path':'data/model_generated_data/backtranslation/wiki103_single_sent_backtranslation.json'},
    'qqp': {'task': 'paraphrase', 'text_a':'question1', 'text_b':'question2', 'label': 'label', 'huggingface': True},
    'qqp_val': {'task': 'paraphrase', 'text_a':'question1', 'text_b':'question2', 'label': 'label', 'huggingface': True},
    'wmt17xxx': {'task': 'wmt', 'text_a': 'ref', 'text_b': 'mt', 'label': 'score','huggingface': False, 'using_hf_api': False, 'using_pandas': True, 'data_path':'data/wmt/wmt17/2017-da.csv' },
    'wmt15': {'task': 'wmt', 'text_a': 'ref', 'text_b': 'mt', 'label': 'score','huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json':True, 'data_path':'data/eval/wmt15_eval.jsonl' },
    'wmt16': {'task': 'wmt', 'text_a': 'ref', 'text_b': 'mt', 'label': 'score','huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json':True, 'data_path':'data/eval/wmt16_eval.jsonl' },
    'wmt17': {'task': 'wmt', 'text_a': 'ref', 'text_b': 'mt', 'label': 'score','huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json':True, 'data_path':'data/eval/wmt17_eval.jsonl' },
    'wmt18': {'task': 'wmt', 'text_a': 'ref', 'text_b': 'mt', 'label': 'score','huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json':True, 'data_path':'data/eval/wmt18_eval.jsonl' },
    'wmt19': {'task': 'wmt', 'text_a': 'ref', 'text_b': 'mt', 'label': 'score','huggingface': False, 'using_hf_api': False, 'using_pandas': False, 'using_json':True, 'data_path':'data/eval/wmt19_eval.jsonl' },
    'squad_v2_new': {'task': 'qa', 'huggingface': True},
    'adversarial_qa': {'task': 'qa', 'huggingface': True},
    'drop': {'task': 'qa', 'huggingface': True},
    'duorc_self': {'task': 'qa', 'huggingface': True},
    'duorc_paraphrase': {'task': 'qa', 'huggingface': True},
    'quoref': {'task': 'qa', 'huggingface': True},
    'hotpot_qa_distractor': {'task': 'qa', 'huggingface': True},
    'hotpot_qa_fullwiki': {'task': 'qa', 'huggingface': True},
    'newsqa': {'task': 'qa',  'using_json': True, 'raw_json': True, 'data_path': 'data/newsqa_raw/combined-newsqa-data-v1.json'},
    'ropes': {'task': 'qa', 'huggingface': True},
    'boolq': {'task': 'qa', 'huggingface': True},
    'eraser_multi_rc': {'task': 'qa', 'huggingface': True},
    'quail': {'task': 'qa', 'huggingface': True},
    'sciq': {'task': 'qa', 'huggingface': True},
    'strategy_qa': {'task': 'qa', 'huggingface': True},
    'gap': {'task': 'coreference', 'huggingface': True},
}


class QA2D():
    def __init__(self, batch_size=32, device='cuda', verbose=True) -> None:
        from transformers import BartTokenizer, BartForConditionalGeneration
        self.tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
        self.model = BartForConditionalGeneration.from_pretrained("MarkS/bart-base-qa2d").to(device)
        self.batch_size = batch_size
        self.device=device
        self.verbose = verbose

    def generate(self, questions: list, answers: list):
        assert len(questions) == len(answers)
        qa_list = []
        for q, a in zip(questions, answers):
            qa_list.append(f"question: {q} answer: {a}")
        output = []
        for qa_pairs in tqdm(
            self.chunks(qa_list, self.batch_size),
            desc="QA to Declarative",
            total=int(len(qa_list)/self.batch_size), 
            disable=(not self.verbose)
        ):
            input_text = qa_pairs
            input_token = self.tokenizer(
                input_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            dec_sents = self.model.generate(
                input_token.input_ids, max_length=512)
            result = self.tokenizer.batch_decode(
                dec_sents, skip_special_tokens=True)
            output.extend(result)

        return output

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


class QAnswering():
    """
    To answer not-answerable questions
    """

    def __init__(self, batch_size=32, device='cuda') -> None:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        self.tokenizer = T5Tokenizer.from_pretrained(
            "valhalla/t5-base-qa-qg-hl")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "valhalla/t5-base-qa-qg-hl").to(device)
        self.batch_size = batch_size
        self.device = device

    def generate(self, questions: list, contexts: list):
        assert len(questions) == len(contexts)
        answers = []
        for qs, cs in tqdm(zip(self.chunks(questions, self.batch_size), self.chunks(contexts, self.batch_size)), desc="Generating Answers for not answerable", total=int(len(questions)/self.batch_size)):
            qc_pairs = []
            assert len(qs) == len(cs)
            for one_q, one_c in zip(qs, cs):
                qc_pairs.append(f"""question: {one_q} context: {one_c}""")
            input_ids = self.tokenizer(
                qc_pairs, padding=True, truncation=True, return_tensors='pt').to(self.device).input_ids
            outputs = self.model.generate(input_ids, max_length=512)
            answers.extend(self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True))

        return answers

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


class MLMGeneratorWithPairedData():
    def __init__(self, corpra: list, device='cuda', batch_size=8, mask_percent=0.25) -> None:
        self.device = device
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.model = transformers.DistilBertForMaskedLM.from_pretrained(
            "distilbert-base-uncased").to(self.device)
        self.mask_percent = mask_percent
        self.batch_size = batch_size

        self.dataset = corpra  # text needs to be noised

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def generate(self):
        sents_output = []
        for examples in tqdm(self.chunks(self.dataset, self.batch_size), total=int(len(self.dataset)/self.batch_size), desc="MLM Generating"):
            sents_to_be_noised = [each for each in examples]
            sents_noised = self.mlm_infiller(sents_to_be_noised)

            sents_output.extend(sents_noised)

        return sents_output

    def mlm_infiller(self, batch):
        """
        input a batch of sentences, list
        """
        masked_batch = []
        masked_batch_ids = []
        for each_sent in batch:
            sent_tokens = self.tokenizer.tokenize(each_sent)
            sent_token_ids = self.tokenizer(each_sent)['input_ids']
            mask_list = sample(list(range(len(sent_tokens))), int(
                self.mask_percent * len(sent_tokens)))
            sent_tokens = [
                each if i not in mask_list else self.tokenizer.mask_token for i, each in enumerate(sent_tokens)]
            masked_batch_ids.append(
                [each if i-1 not in mask_list else self.tokenizer.mask_token_id for i, each in enumerate(sent_token_ids)])
            masked_batch.append(' '.join(sent_tokens))

        inputs = self.tokenizer(
            masked_batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        infill_tokens = []
        for i in range(len(masked_batch)):
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[
                i].nonzero(as_tuple=True)[0]

            predicted_token_id = logits[i, mask_token_index].argmax(axis=-1)
            infill_tokens.append(predicted_token_id)

        infilled_sent = []
        for masked_sent_ids, infill_token in zip(masked_batch_ids, infill_tokens):
            for infill_one_token in infill_token:
                for i, each_id in enumerate(masked_sent_ids):
                    if each_id == self.tokenizer.mask_token_id:
                        masked_sent_ids[i] = infill_one_token
                        break
            infilled_sent.append(self.tokenizer.decode(
                masked_sent_ids, skip_special_tokens=True))

        return infilled_sent


class ExtractiveSummarizationGenerator():
    def __init__(self) -> None:
        pass

    def generate(self, texts):
        '''
        texts: list of string
        '''
        from summa.summarizer import summarize

        summaries = []
        for text in tqdm(texts, desc="Extracting Summary"):
            for prop in range(1, 20):
                summ = summarize(text, ratio=prop/20.)
                if len(summ) > 0:
                    break
            summaries.append(summ)

        return summaries


class DataGenerator():
    def __init__(self, dataset_names) -> None:
        self.dataset_names = dataset_names
        self.datasets = dict()
        self.t5_qa = None
        self.t5_tokenizer = None

        self.load_dataset_from_huggingface()

    def load_dataset_from_huggingface(self):
        for each_dataset in self.dataset_names:
            if DATASET_CONFIG[each_dataset].get('huggingface'):
                self.datasets[each_dataset] = load_dataset(
                    *DATASET_HUGGINGFACE[each_dataset][:-1])[DATASET_HUGGINGFACE[each_dataset][-1]]
            elif DATASET_CONFIG[each_dataset].get('using_hf_api'):
                self.datasets[each_dataset] = load_dataset(
                    *DATASET_HUGGINGFACE[each_dataset][:-1], data_dir=DATASET_CONFIG[each_dataset]['data_dir'])[DATASET_HUGGINGFACE[each_dataset][-1]]
            elif DATASET_CONFIG[each_dataset].get('using_pandas'):
                if DATASET_CONFIG[each_dataset]['data_path'].split('.')[-1] == 'tsv':
                    self.datasets[each_dataset] = pd.read_csv(
                        DATASET_CONFIG[each_dataset]['data_path'], sep='\t')
                elif DATASET_CONFIG[each_dataset]['data_path'].split('.')[-1] == 'csv':
                    self.datasets[each_dataset] = pd.read_csv(
                        DATASET_CONFIG[each_dataset]['data_path'])
            elif DATASET_CONFIG[each_dataset].get('using_json'):
                self.datasets[each_dataset] = []
                if DATASET_CONFIG[each_dataset].get('raw_json'):
                    with open(DATASET_CONFIG[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                        self.datasets[each_dataset] = json.load(f)
                else:
                    try:
                        json_file = json.load(
                            open(DATASET_CONFIG[each_dataset]['data_path'], 'r', encoding='utf8'))
                        for example in json_file:
                            self.datasets[each_dataset].append(example)
                    except:
                        with open(DATASET_CONFIG[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                            for example in f:
                                self.datasets[each_dataset].append(
                                    json.loads(example))
            else:
                error('unable to locate raw dataset...')

    def process_squad(self):
        from rake_nltk import Rake
        r = Rake()
        topk = 5
        threshold = 0.6

        output = []
        label = -1
        for example in tqdm(self.datasets['squad'], desc=f'Constructing squad'):
            text_a = example[DATASET_CONFIG['squad']['text_a']]
            question = example[DATASET_CONFIG['squad']['text_b'][0]]
            answer = example[DATASET_CONFIG['squad']
                             ['text_b'][1]]['text']  # a list
            text_b = [question+' '+answer_ele for answer_ele in answer]
            text_c = []

            r.extract_keywords_from_text(text_a)
            keywords_in_context = r.get_ranked_phrases()[:topk]
            for each_keyword in keywords_in_context:
                # then it is an incorrect answer
                if sentence_bleu([answer_ele.lower().split() for answer_ele in answer], each_keyword.split(), weights=(0.33, 0.33, 0.33)) < threshold:
                    text_c.append(question+' '+each_keyword)

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_squad_v2(self):
        # first collect answerable items
        not_answerable_contexts = []
        not_answerable_questions = []
        not_answerable_answers = []

        answerable_contexts = []
        answerable_questions = []
        answerable_answers = []

        qa_generator = QAnswering(batch_size=32, device='cuda')
        qa2d_generator = QA2D(batch_size=32, device='cuda')

        for example in tqdm(self.datasets['squad_v2'], desc=f'Collecting (not)answerable examples'):
            if len(example['answers']['text']) == 0:
                not_answerable_contexts.append(example['context'])
                not_answerable_questions.append(example['question'])
            else:
                answerable_contexts.append(example['context'])
                answerable_questions.append(example['question'])
                answerable_answers.append(example['answers']['text'][0])

        not_answerable_answers = qa_generator.generate(
            not_answerable_questions, not_answerable_contexts)
        answerable_declarative_sents = qa2d_generator.generate(
            answerable_questions, answerable_answers)
        not_answerable_declarative_sents = qa2d_generator.generate(
            not_answerable_questions, not_answerable_answers)

        output = []
        for i, dec_sent in enumerate(answerable_declarative_sents):
            output.append({
                'text_a': answerable_contexts[i],
                'text_b': [dec_sent],
                'text_c': [],
                'label': 1
            })

        for i, dec_sent in enumerate(not_answerable_declarative_sents):
            output.append({
                'text_a': not_answerable_contexts[i],
                'text_b': [dec_sent],
                'text_c': [],
                'label': 0
            })

        return output

    def process_race(self):
        qa2d_generator = QA2D(batch_size=32, device='cuda')
        option_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        output = []

        correct_context = []
        correct_question = []
        correct_answer = []

        wrong_context = []
        wrong_question = []
        wrong_answer = []

        for example in tqdm(self.datasets['race'], desc=f'Constructing race'):
            text_a = example[DATASET_CONFIG['race']['text_a']]
            label = -1
            question = example[DATASET_CONFIG['race']['text_b'][0]]
            if "_" in question:
                answer_id = option_dict[example[DATASET_CONFIG['race']['label']]]
                for i, options in enumerate(example[DATASET_CONFIG['race']['text_b'][1]]):
                    if i == answer_id:
                        output.append({
                            'text_a': text_a,
                            'text_b': [' '.join(question.replace("_", " "+options+" ").split())],
                            'text_c': [],
                            'label': 1
                        })
                    else:
                        output.append({
                            'text_a': text_a,
                            'text_b': [' '.join(question.replace("_", " "+options+" ").split())],
                            'text_c': [],
                            'label': 0
                        })
            else:
                answer_id = option_dict[example[DATASET_CONFIG['race']['label']]]
                for i, options in enumerate(example[DATASET_CONFIG['race']['text_b'][1]]):
                    if i == answer_id:
                        output.append({
                                'text_a': text_a,
                                'text_b': [question],
                                'text_c': [options],
                                'label': 1
                            })
                    else:
                        output.append({
                                'text_a': text_a,
                                'text_b': [question],
                                'text_c': [options],
                                'label': 0
                            })

        return output

    def process_race_val(self):
        qa2d_generator = QA2D(batch_size=32, device='cuda')
        option_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        output = []

        correct_context = []
        correct_question = []
        correct_answer = []

        wrong_context = []
        wrong_question = []
        wrong_answer = []

        for example in tqdm(self.datasets['race_val'], desc=f'Constructing race_val'):
            text_a = example[DATASET_CONFIG['race_val']['text_a']]
            label = -1
            question = example[DATASET_CONFIG['race_val']['text_b'][0]]
            if "_" in question:
                answer_id = option_dict[example[DATASET_CONFIG['race_val']['label']]]
                for i, options in enumerate(example[DATASET_CONFIG['race_val']['text_b'][1]]):
                    if i == answer_id:
                        output.append({
                            'text_a': text_a,
                            'text_b': [' '.join(question.replace("_", " "+options+" ").split())],
                            'text_c': [],
                            'label': 1
                        })
                    else:
                        output.append({
                            'text_a': text_a,
                            'text_b': [' '.join(question.replace("_", " "+options+" ").split())],
                            'text_c': [],
                            'label': 0
                        })
            else:
                answer_id = option_dict[example[DATASET_CONFIG['race_val']['label']]]
                for i, options in enumerate(example[DATASET_CONFIG['race_val']['text_b'][1]]):
                    if i == answer_id:
                        correct_context.append(text_a)
                        correct_question.append(question)
                        correct_answer.append(options)
                    else:
                        wrong_context.append(text_a)
                        wrong_question.append(question)
                        wrong_answer.append(options)

        correct_declarative = qa2d_generator.generate(
            correct_question, correct_answer)
        wrong_declarative = qa2d_generator.generate(
            wrong_question, wrong_answer)
        assert len(correct_context) == len(correct_declarative)
        assert len(wrong_context) == len(wrong_declarative)
        for context, dec in zip(correct_context, correct_declarative):
            output.append({
                'text_a': context,
                'text_b': [dec],
                'text_c': [],
                'label': 1
            })

        for context, dec in zip(wrong_context, wrong_declarative):
            output.append({
                'text_a': context,
                'text_b': [dec],
                'text_c': [],
                'label': 0
            })

        return output

    def process_race_test(self):
        option_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        output = []
        for example in tqdm(self.datasets['race_test'], desc=f'Constructing race_test'):
            text_a = example[DATASET_CONFIG['race_test']['text_a']]
            text_b = []  # pos
            text_c = []  # neg
            label = -1
            question = example[DATASET_CONFIG['race_test']['text_b'][0]]
            if "_" in question:
                answer_id = option_dict[example[DATASET_CONFIG['race_test']['label']]]
                for i, options in enumerate(example[DATASET_CONFIG['race_test']['text_b'][1]]):
                    if i == answer_id:
                        text_b.append(' '.join(question.replace(
                            "_", " "+options+" ").split()))
                    else:
                        text_c.append(' '.join(question.replace(
                            "_", " "+options+" ").split()))
            else:
                answer_id = option_dict[example[DATASET_CONFIG['race_test']['label']]]
                for i, options in enumerate(example[DATASET_CONFIG['race_test']['text_b'][1]]):
                    if i == answer_id:
                        text_b.append(question+" "+options+" ")
                    else:
                        text_c.append(question+" "+options+" ")

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_xsum(self):
        '''
        text_a: raw_text
        text_b: raw_summary + ***extractive summ*** removed
        text_c: cliff xsum + DistillBERT from raw_text_b + ***DistillBERT from extractive summ text_b***
        '''
        output = []

        gold_summary = [example[DATASET_CONFIG['xsum']['text_b']]
                        for example in self.datasets['xsum']]
        ext_summarizer = ExtractiveSummarizationGenerator()
        extracted_summ = ext_summarizer.generate(
            [example[DATASET_CONFIG['xsum']['text_a']] for example in self.datasets['xsum']])

        mlm_hallucinator = MLMGeneratorWithPairedData(
            corpra=gold_summary, device='cuda:0', batch_size=64, mask_percent=0.25)
        gold_summary_hallucinated = mlm_hallucinator.generate()

        mlm_hallucinator = MLMGeneratorWithPairedData(
            corpra=extracted_summ, device='cuda:0', batch_size=64, mask_percent=0.25)
        extracted_summ_hallucinated = mlm_hallucinator.generate()

        assert len(self.datasets['xsum']) == len(gold_summary_hallucinated) and len(
            self.datasets['xsum']) == len(extracted_summ_hallucinated)

        for i, example in tqdm(enumerate(self.datasets['xsum']), desc="Constructing xsum", total=len(self.datasets['xsum'])):
            text_a = example[DATASET_CONFIG['xsum']['text_a']]
            text_b = [gold_summary[i], extracted_summ[i]]
            text_c = [gold_summary_hallucinated[i],
                      extracted_summ_hallucinated[i]]
            label = -1

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_cnndm(self):
        '''
        text_a: raw_text
        text_b: raw_summary + ***extractive summ*** removed
        text_c: DistillBERT from raw_text_b + ***DistillBERT from extractive summ text_b***
        '''
        # interpretation of fairseq-generate output: https://github.com/facebookresearch/fairseq/issues/3000
        output = []

        gold_summary = [example[DATASET_CONFIG['cnndm']['text_b']]
                        for example in self.datasets['cnndm']]
        ext_summarizer = ExtractiveSummarizationGenerator()
        extracted_summ = ext_summarizer.generate(
            [example[DATASET_CONFIG['cnndm']['text_a']] for example in self.datasets['cnndm']])

        mlm_hallucinator = MLMGeneratorWithPairedData(
            corpra=gold_summary, device='cuda:0', batch_size=64, mask_percent=0.25)
        gold_summary_hallucinated = mlm_hallucinator.generate()

        mlm_hallucinator = MLMGeneratorWithPairedData(
            corpra=extracted_summ, device='cuda:0', batch_size=64, mask_percent=0.25)
        extracted_summ_hallucinated = mlm_hallucinator.generate()

        assert len(self.datasets['cnndm']) == len(gold_summary_hallucinated) and len(
            self.datasets['cnndm']) == len(extracted_summ_hallucinated)

        for i, example in tqdm(enumerate(self.datasets['cnndm']), desc="Constructing cnndm", total=len(self.datasets['cnndm'])):
            text_a = example[DATASET_CONFIG['cnndm']['text_a']]
            text_b = [gold_summary[i], extracted_summ[i]]
            text_c = [gold_summary_hallucinated[i],
                      extracted_summ_hallucinated[i]]
            label = -1

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_wikihow(self):
        '''
        text_a: raw_text
        text_b: raw_summary + ***extractive summ*** removed
        text_c: DistillBERT from raw_text_b + ***DistillBERT from extractive summ text_b***
        '''
        # interpretation of fairseq-generate output: https://github.com/facebookresearch/fairseq/issues/3000
        output = []

        gold_summary = [example[DATASET_CONFIG['wikihow']['text_b']]
                        for example in self.datasets['wikihow']]
        ext_summarizer = ExtractiveSummarizationGenerator()
        extracted_summ = ext_summarizer.generate(
            [example[DATASET_CONFIG['wikihow']['text_a']] for example in self.datasets['wikihow']])

        mlm_hallucinator = MLMGeneratorWithPairedData(
            corpra=gold_summary, device='cuda:0', batch_size=64, mask_percent=0.25)
        gold_summary_hallucinated = mlm_hallucinator.generate()

        mlm_hallucinator = MLMGeneratorWithPairedData(
            corpra=extracted_summ, device='cuda:0', batch_size=64, mask_percent=0.25)
        extracted_summ_hallucinated = mlm_hallucinator.generate()

        assert len(self.datasets['wikihow']) == len(gold_summary_hallucinated) and len(
            self.datasets['wikihow']) == len(extracted_summ_hallucinated)

        for i, example in tqdm(enumerate(self.datasets['wikihow']), desc="Constructing wikihow", total=len(self.datasets['wikihow'])):
            text_a = example[DATASET_CONFIG['wikihow']['text_a']]
            text_b = [gold_summary[i], extracted_summ[i]]
            text_c = [gold_summary_hallucinated[i],
                      extracted_summ_hallucinated[i]]
            label = -1

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_wiki103(self):
        output = []

        paraphrases = [example[DATASET_CONFIG['wiki103']['text_b']]
                       for example in self.datasets['wiki103']]
        mlm_hallucinator = MLMGeneratorWithPairedData(
            corpra=paraphrases, device='cuda:3', batch_size=64, mask_percent=0.25)
        paraphrase_hallucinated = mlm_hallucinator.generate()

        assert len(self.datasets['wiki103']) == len(paraphrase_hallucinated)

        for i, example in tqdm(enumerate(self.datasets['wiki103']), desc=f'Constructing wiki103'):
            output.append({
                'text_a': example[DATASET_CONFIG['wiki103']['text_a']],
                'text_b': [example[DATASET_CONFIG['wiki103']['text_b']]],
                'text_c': [],
                'label': 1
            })
            output.append({
                'text_a': example[DATASET_CONFIG['wiki103']['text_a']],
                'text_b': [paraphrase_hallucinated[i]],
                'text_c': [],
                'label': 0
            })

        return output

    def process_mnli(self):
        output = []
        for example in tqdm(self.datasets['mnli'], desc=f'Constructing mnli'):
            text_a = example[DATASET_CONFIG['mnli']['text_a']]
            text_b = [example[DATASET_CONFIG['mnli']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['mnli']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_nli_fever(self):
        output = []
        for example in tqdm(self.datasets['nli_fever'], desc=f'Constructing nli_fever'):
            text_a = example[DATASET_CONFIG['nli_fever']['text_a']]
            text_b = [example[DATASET_CONFIG['nli_fever']['text_b']]]
            text_c = []
            raw_label = example[DATASET_CONFIG['nli_fever']['label']]
            if raw_label == 'SUPPORTS':  # convert to nli style label
                label = 0
            elif raw_label == 'REFUTES':
                label = 2
            else:
                label = 1

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_doc_nli(self):
        output = []
        for example in tqdm(self.datasets['doc_nli'], desc=f'Constructing doc_nli'):
            text_a = example[DATASET_CONFIG['doc_nli']['text_a']]
            text_b = [example[DATASET_CONFIG['doc_nli']['text_b']]]
            text_c = []
            raw_label = example[DATASET_CONFIG['doc_nli']['label']]
            if raw_label == 'entailment':  # convert to paraphrase style label
                label = 1
            else:
                label = 0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_anli_r1(self):
        output = []
        for example in tqdm(self.datasets['anli_r1'], desc=f'Constructing anli_r1'):
            text_a = example[DATASET_CONFIG['anli_r1']['text_a']]
            text_b = [example[DATASET_CONFIG['anli_r1']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['anli_r1']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_anli_r2(self):
        output = []
        for example in tqdm(self.datasets['anli_r2'], desc=f'Constructing anli_r2'):
            text_a = example[DATASET_CONFIG['anli_r2']['text_a']]
            text_b = [example[DATASET_CONFIG['anli_r2']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['anli_r2']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_anli_r3(self):
        output = []
        for example in tqdm(self.datasets['anli_r3'], desc=f'Constructing anli_r3'):
            text_a = example[DATASET_CONFIG['anli_r3']['text_a']]
            text_b = [example[DATASET_CONFIG['anli_r3']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['anli_r3']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_snli(self):
        output = []
        for example in tqdm(self.datasets['snli'], desc=f'Constructing snli'):
            text_a = example[DATASET_CONFIG['snli']['text_a']]
            text_b = [example[DATASET_CONFIG['snli']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['snli']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_paws(self):
        output = []
        for example in tqdm(self.datasets['paws'], desc=f'Constructing paws'):
            text_a = example[DATASET_CONFIG['paws']['text_a']]
            text_b = [example[DATASET_CONFIG['paws']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['paws']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_vitaminc(self):
        output = []
        for example in tqdm(self.datasets['vitaminc'], desc=f'Constructing vitaminc'):
            text_a = example[DATASET_CONFIG['vitaminc']['text_a']]
            text_b = [example[DATASET_CONFIG['vitaminc']['text_b']]]
            text_c = []
            raw_label = example[DATASET_CONFIG['vitaminc']['label']]
            if raw_label == 'SUPPORTS':  # convert to nli style label
                label = 0
            elif raw_label == 'REFUTES':
                label = 2
            else:
                label = 1

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_stsb(self):
        output = []
        for example in tqdm(self.datasets['stsb'], desc=f'Constructing stsb'):
            text_a = example[DATASET_CONFIG['stsb']['text_a']]
            text_b = [example[DATASET_CONFIG['stsb']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['stsb']['label']] / 5.0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_sick(self):
        output = []
        for example in tqdm(self.datasets['sick'], desc=f'Constructing sick'):
            text_a = example[DATASET_CONFIG['sick']['text_a']]
            text_b = [example[DATASET_CONFIG['sick']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['sick']['label']] / 5.0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_mrpc(self):
        output = []
        for example in tqdm(self.datasets['mrpc'], desc=f'Constructing mrpc'):
            text_a = example[DATASET_CONFIG['mrpc']['text_a']]
            text_b = [example[DATASET_CONFIG['mrpc']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['mrpc']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_mrpc_val(self):
        output = []
        for example in tqdm(self.datasets['mrpc_val'], desc=f'Constructing mrpc_val'):
            text_a = example[DATASET_CONFIG['mrpc_val']['text_a']]
            text_b = [example[DATASET_CONFIG['mrpc_val']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['mrpc_val']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_paws_val(self):
        output = []
        for example in tqdm(self.datasets['paws_val'], desc=f'Constructing paws_val'):
            text_a = example[DATASET_CONFIG['paws_val']['text_a']]
            text_b = [example[DATASET_CONFIG['paws_val']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['paws_val']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_paws_unlabeled(self):
        output = []
        for example in tqdm(self.datasets['paws_unlabeled'], desc=f'Constructing paws_unlabeled'):
            text_a = example[DATASET_CONFIG['paws_unlabeled']['text_a']]
            text_b = [example[DATASET_CONFIG['paws_unlabeled']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['paws_unlabeled']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_qqp(self):
        output = []
        for example in tqdm(self.datasets['qqp'], desc=f'Constructing qqp'):
            text_a = example[DATASET_CONFIG['qqp']['text_a']]
            text_b = [example[DATASET_CONFIG['qqp']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['qqp']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_qqp_val(self):
        output = []
        for example in tqdm(self.datasets['qqp_val'], desc=f'Constructing qqp_val'):
            text_a = example[DATASET_CONFIG['qqp_val']['text_a']]
            text_b = [example[DATASET_CONFIG['qqp_val']['text_b']]]
            text_c = []
            label = example[DATASET_CONFIG['qqp_val']['label']]

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_msmarco(self):
        qa2d_generator = QA2D(batch_size=32, device='cuda')
        output = []
        correct_contexts = []
        correct_questions = []
        correct_answers = []

        wrong_contexts = []
        wrong_questions = []
        wrong_answers = []

        filtered_examples = []
        questions = []
        answers = []
        declaratives = []

        for example in tqdm(self.datasets['msmarco'], desc=f'Collecting msmarco'):
            if sum(example['passages']['is_selected']) > 0:  # has answer
                questions.append(example['query'])
                answers.append(example['answers'][0] if len(
                    example['wellFormedAnswers']) == 0 else example['wellFormedAnswers'][0])
                filtered_examples.append(example)
        
        for example in filtered_examples:
            for i, is_selected in enumerate(example['passages']['is_selected']):
                if is_selected == 1:
                    output.append({
                        'text_a': example['passages']['passage_text'][i],
                        'text_b': [example['query']],
                        'text_c': [],
                        'label': 1
                    }
                    )
                else:
                    output.append({
                        'text_a': example['passages']['passage_text'][i],
                        'text_b': [example['query']],
                        'text_c': [],
                        'label': 0
                    }
                    )
        return output

    def process_paws_qqp(self):
        output = []

        for i in range(len(self.datasets['paws_qqp'])):
            text_a = self.datasets['paws_qqp'].iloc[i]['sentence1'][2:-1]
            text_b = [self.datasets['paws_qqp'].iloc[i]['sentence2'][2:-1]]
            text_c = []
            label = self.datasets['paws_qqp'].iloc[i]['label']

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': int(label)
            })

        return output

    def process_wmt15(self):
        output = []

        for example in self.datasets['wmt15']:
            text_a = example['reference']
            text_b = [example['candidate']]
            text_c = []
            label = example['score']

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_wmt16(self):
        output = []

        for example in self.datasets['wmt16']:
            text_a = example['reference']
            text_b = [example['candidate']]
            text_c = []
            label = example['score']

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_wmt17(self):

        output = []

        for example in self.datasets['wmt17']:
            text_a = example['reference']
            text_b = [example['candidate']]
            text_c = []
            label = example['score']

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_wmt18(self):
        output = []

        for example in self.datasets['wmt18']:
            text_a = example['reference']
            text_b = [example['candidate']]
            text_c = []
            label = example['score']

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_wmt19(self):
        output = []

        for example in self.datasets['wmt19']:
            text_a = example['reference']
            text_b = [example['candidate']]
            text_c = []
            label = example['score']

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output
    
    def process_boolq(self):
        output = []

        for example in self.datasets['boolq']:
            text_a = example['passage']
            text_b = [example['question']]
            text_c = ["Yes." if example['answer'] else "No."]
            label = 1

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

            text_a = example['passage']
            text_b = [example['question']]
            text_c = ["Yes." if not example['answer'] else "No."]
            label = 0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output
    
    def process_eraser_multi_rc(self):
        output = []

        for example in self.datasets['eraser_multi_rc']:
            text_a = example['passage']
            text_b = [example['query_and_answer'].replace("|", "")]
            text_c = []
            label = int(example['label'])

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output
    
    def process_quail(self):
        output = []

        for example in self.datasets['quail']:
            for i, ans in enumerate(example['answers']):
                text_a = example['context']
                text_b = [example['question']]
                text_c = [ans]
                label = 1 if i == example['correct_answer_id'] else 0

                output.append({
                    'text_a': text_a,
                    'text_b': text_b,
                    'text_c': text_c,
                    'label': label
                })

        return output
    
    def process_sciq(self):
        output = []

        for example in self.datasets['sciq']:
            text_a = example['support']

            output.append({
                'text_a': text_a,
                'text_b': [example['question']],
                'text_c': [example['distractor1']],
                'label': 0
            })
            output.append({
                'text_a': text_a,
                'text_b': [example['question']],
                'text_c': [example['distractor2']],
                'label': 0
            })
            output.append({
                'text_a': text_a,
                'text_b': [example['question']],
                'text_c': [example['distractor3']],
                'label': 0
            })
            output.append({
                'text_a': text_a,
                'text_b': [example['question']],
                'text_c': [example['correct_answer']],
                'label': 1
            })

        return output
    
    def process_strategy_qa(self):
        output = []

        for example in self.datasets['strategy_qa']:
            text_a = ' '.join(example['facts'])
            text_b = [example['question']]
            text_c = ["Yes." if example['answer'] else "No."]
            label = 1

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

            text_a = ' '.join(example['facts'])
            text_b = [example['question']]
            text_c = ["Yes." if not example['answer'] else "No."]
            label = 0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def process_gap(self):
        output = []

        for example in self.datasets['gap']:
            text_a = example['Text']
            text_b = [example['Text'][:example['Pronoun-offset']]+example['A']+example['Text'][(example['Pronoun-offset']+len(example['Pronoun'])):]]
            text_c = []
            label = 1 if example['A-coref'] else 0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

            text_a = example['Text']
            text_b = [example['Text'][:example['Pronoun-offset']]+example['B']+example['Text'][(example['Pronoun-offset']+len(example['Pronoun'])):]]
            text_c = []
            label = 1 if example['B-coref'] else 0

            output.append({
                'text_a': text_a,
                'text_b': text_b,
                'text_c': text_c,
                'label': label
            })

        return output

    def init_qa_t5(self):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        if self.t5_qa is None:
            self.t5_tokenizer = T5Tokenizer.from_pretrained(
                "t5-base", model_max_length=800)
            self.t5_qa = T5ForConditionalGeneration.from_pretrained("t5-base")
            self.t5_qa.to('cuda:1')
            self.t5_qa.eval()

    @staticmethod
    def mask_answer(context, answers):
        answers = sorted(answers, key=len, reverse=True)
        for answer in answers:
            pattern = f'(?<![\w\\-\u2013]){re.escape(answer)}(?![\w\\-\u2013])'
            context = re.sub(pattern, '', context, flags=re.IGNORECASE)
        return context

    def generate_fake_answer(self, context, question, answers):
        self.init_qa_t5()

        context_no_answer = self.mask_answer(context, answers)

        input_ids = self.t5_tokenizer(
            f'question: {question} context: {context_no_answer}',
            return_tensors="pt",
            truncation='only_first'
        ).input_ids.to(self.t5_qa.device)

        outputs = self.t5_qa.generate(
            input_ids,
            max_new_tokens=40,
            remove_invalid_values=True
        )

        return self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def negative_sample_qa(self, samples, negative_sample_no_ans_only=True):
        outputs = []
        for context, question, answers in samples:
            if answers:
                outputs.append({
                    'text_a': context,
                    'text_b': [question],
                    'text_c': answers,
                    'label': 1
                })
            if not answers or not negative_sample_no_ans_only:
                fake_answer = self.generate_fake_answer(
                    context, question, answers)
                outputs.append({
                    'text_a': context,
                    'text_b': [question],
                    'text_c': [fake_answer],
                    'label': 0
                })

        return outputs

    def process_squad_v2_new(self):
        samples = (
            (sample['context'], sample['question'], sample['answers']['text'])
            for sample in tqdm(self.datasets['squad_v2_new'], desc=f'squad_v2_new')
        )
        return self.negative_sample_qa(samples)

    def process_adversarial_qa(self):
        samples = (
            (sample['context'], sample['question'], sample['answers']['text'])
            for sample in tqdm(self.datasets['adversarial_qa'], desc=f'adversarial_qa')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def process_drop(self):
        samples = (
            (sample['passage'], sample['question'],
             sample['answers_spans']['spans'])
            for sample in tqdm(self.datasets['drop'], desc=f'drop')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def process_duorc_self(self):
        samples = (
            (sample['plot'], sample['question'],
             sample['answers'])
            for sample in tqdm(self.datasets['duorc_self'], desc=f'duorc_self')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def process_duorc_paraphrase(self):
        samples = (
            (sample['plot'], sample['question'],
             sample['answers'])
            for sample in tqdm(self.datasets['duorc_paraphrase'], desc=f'duorc_paraphrase')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def process_quoref(self):
        samples = (
            (sample['context'], sample['question'], sample['answers']['text'])
            for sample in tqdm(self.datasets['quoref'], desc=f'quoref')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    @staticmethod
    def prepare_hotpot_qa_samples(dateset):
        for sample in dateset:
            question = sample['question']
            answer = sample['answer']
            supporting_docs = set(sample['supporting_facts']['title'])
            irrelevant_docs = []
            context_paragraphs = []
            for title, setences in zip(sample['context']['title'], sample['context']['sentences']):
                doc = ''.join(setences)
                if title in supporting_docs:
                    context_paragraphs.append(doc)
                else:
                    irrelevant_docs.append(doc)
            # Add some irrelevant documents
            if irrelevant_docs and len(context_paragraphs) < 4:
                context_paragraphs.append(random.choice(irrelevant_docs))
            random.shuffle(context_paragraphs)
            yield '\n'.join(context_paragraphs), question, [answer]

    def process_hotpot_qa_distractor(self):
        samples = self.prepare_hotpot_qa_samples(
            tqdm(self.datasets['hotpot_qa_distractor'],
                 desc=f'hotpot_qa_distractor')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def process_hotpot_qa_fullwiki(self):
        samples = self.prepare_hotpot_qa_samples(
            tqdm(self.datasets['hotpot_qa_fullwiki'],
                 desc=f'hotpot_qa_fullwiki')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def process_newsqa(self):
        def get_samples(dataset):
            for story in tqdm(dataset['data'], desc='newsqa'):
                if story['type'] != 'train':
                    continue
                context = story['text']
                for question in story['questions']:
                    if question.get('isQuestionBad', 0.) > 0.2:
                        continue
                    answers = []
                    if 's' in question['consensus']:
                        start = question['consensus']['s']
                        end = question['consensus']['e']
                        answers.append(context[start:end].strip())
                    yield context, question['q'], answers
        samples = get_samples(self.datasets['newsqa'])
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def process_ropes(self):
        samples = (
            (
                sample['situation'] + ' ' + sample['background'],
                sample['question'], sample['answers']['text']
            )
            for sample in tqdm(self.datasets['ropes'], desc=f'ropes')
        )
        return self.negative_sample_qa(samples, negative_sample_no_ans_only=False)

    def generate(self):
        for each_dataset in self.datasets:
            with open(f'./data/training/{each_dataset}.json', 'w', encoding='utf8') as outfile:
                outfile.write("")
        for each_dataset in self.datasets:
            outputs = eval(f'self.process_{each_dataset}()')

            for each_output in outputs:
                dict_write_to_file = {
                    'task': DATASET_CONFIG[each_dataset]['task'],
                    'text_a': each_output['text_a'],  # string
                    # list of positive examples
                    'text_b': each_output['text_b'],
                    # list of negative examples
                    'text_c': each_output['text_c'],
                    # original label, if -1 only has positive pairs and negative pairs
                    'orig_label': each_output['label']
                }
                with open(f'./data/training/{each_dataset}.json', 'a', encoding='utf8') as outfile:
                    json.dump(dict_write_to_file, outfile, ensure_ascii=False)
                    outfile.write('\n')


if __name__ == "__main__":
    random.seed(42)
    gen = DataGenerator(list(DATASET_CONFIG.keys()))
    gen.generate()
