from logging import warning
from datasets import load_dataset
from alignscore.inference import Inferencer
import numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef
import pandas as pd
import torch
import json
import pickle
import os

HUGGINGFACE_DATASETS = {
    'stsb': ['glue', 'stsb', 'validation'],
    'mrpc': ['glue', 'mrpc', 'test'],
    'axb': ['super_glue', 'axb', 'test'],
    'axg': ['super_glue', 'axg', 'test'],
    'cb': ['super_glue', 'cb', 'validation'],
    'rte': ['super_glue', 'rte', 'validation'],
    'wnli': ['SetFit/wnli', 'validation'],
    'paws': ['paws', 'labeled_final', 'test'],
    'mnli_matched': ['multi_nli', 'validation_matched'],
    'mnli_mismatched': ['multi_nli', 'validation_mismatched'],
    'nli_fever': ['pietrolesci/nli_fever', 'dev'],
    'doc_nli': ['saattrupdan/doc-nli', 'test'],
    'sem_eval': ['sem_eval_2014_task_1', 'test'],
    'sick': ['sick', 'default', 'test'],
    'race_m': ['race', 'middle', 'test'],
    'race_h': ['race', 'high', 'test'],
    'boolq': ['boolq', 'validation'],
    'anli_1': ['anli', 'test_r1'],
    'anli_2': ['anli', 'test_r2'],
    'anli_3': ['anli', 'test_r3'],
    'snli': ['snli', 'test'],
    'vitaminc': ['tals/vitaminc', 'test'],
    'qqp': ['glue', 'qqp', 'validation'],
    # below are tasks from https://arxiv.org/pdf/2104.14690.pdf
    'sst2': ['SetFit/sst2', 'test'],
    # can't find MR
    'cr': ['SetFit/SentEval-CR', 'test'],
    # can't find MPQA
    'subj': ['SetFit/subj', 'test'],
    # can't find OS
    'imdb': ['SetFit/imdb', 'test'], # note: I can't confirm if this is the same dataset used in that paper
                                     # The original dataset is no longer accessiable
    'cola': ['glue', 'cola', 'validation'],
    'yelp_efl': ['SetFit/yelp_review_full', 'test'],
    'ag_news': ['SetFit/ag_news', 'test'],
    'trec': ['SetFit/TREC-QC', 'test',],
    'dream': ['dream', 'test'],
    'quartz': ['quartz', 'test'],
    'eraser_multi_rc': ['eraser_multi_rc', 'test'],
    'quail': ['quail', 'challenge'],
    'sciq': ['sciq', 'test'],
    'gap': ['gap', 'test'],
    'qnli': ['glue', 'qnli', 'validation']
}

PICKLE_DATASETS = [
    'newsroom',
    'rank19',
    'bagel',
    'sfhot',
    'sfres'
]

ALL_TASKS = { # enumerate all possible tasks
    'stsb': 0, ### using which output: regression, binary, tri-label
    'sick': 0,
    'race_m': 1,
    'race_h': 1,
    'boolq': 1,
    'anli_1': 2,
    'anli_2': 2,
    'anli_3': 2,
    'snli': 2,
    'vitaminc': 2,
    'mrpc': 1,
    'paws': 1,
    'mnli_matched': 2,
    'mnli_mismatched': 2,
    'sem_eval': 1,
    'summeval': 1,
    'qags_xsum': 1,
    'qags_cnndm': 1,
    'frank': 1,
    'xsumfaith': 1,
    'samsum': 1,
    'yelp': 1,
    'persona_chat': 1,
    'topical_chat': 1,
    'paws_qqp': 1,
    'qqp': 1,
    'newsroom': 1,
    'rank19': 1,
    'bagel': 1,
    'sfhot': 1,
    'sfres': 1,
    'wmt17': 0,
    'wmt18': 0,
    'wmt19': 0,
    'sst2': 1,
    'cr': 1,
    'subj': 1,
    'imdb': 1,
    'cola': 1,
    'yelp_efl': 1,
    'ag_news': 1,
    'trec': 1,
    'axb': 1,
    'axg': 1,
    'cb': 2,
    'rte': 2,
    'wnli': 2,
    'dream': 1,
    'quartz': 1,
    'nli_fever': 2,
    'doc_nli': 1,
    'eraser_multi_rc': 1,
    'quail': 1,
    'sciq': 1,
    'gap': 1,
    'qnli': 1
}

FEW_SHOT_N = 8
FEW_SHOT_SEEDS = [30247, 38252, 29050, 1091, 35554, 25309, 79319, 35079, 35256, 46744]

class Evaluator():
    def __init__(self, eval_tasks, align_func, save_all_tables=False, clean_data=True) -> None:
        self.align_func = align_func
        self.eval_tasks = eval_tasks # ['stsb', 'paws', ...]
        self.result_save_name = "Default_result_name"
        self.result_tables = []
        self.result_dicts = []
        self.clean_data = clean_data
        self.init_eval_dataset()

        self.should_save_all_tables = save_all_tables
        warning(f"Saving the result is: {self.should_save_all_tables}")
    
    def init_eval_dataset(self):
        self.dataset = dict()
        for eval_task in self.eval_tasks:
            if eval_task in HUGGINGFACE_DATASETS:
                if len(HUGGINGFACE_DATASETS[eval_task]) == 3:
                    self.dataset[eval_task] = load_dataset(HUGGINGFACE_DATASETS[eval_task][0], HUGGINGFACE_DATASETS[eval_task][1])[HUGGINGFACE_DATASETS[eval_task][2]]
                elif len(HUGGINGFACE_DATASETS[eval_task]) == 2:
                    if isinstance(HUGGINGFACE_DATASETS[eval_task][1], tuple):
                        dataset = load_dataset(HUGGINGFACE_DATASETS[eval_task][0])
                        self.dataset[eval_task] = {split:dataset[split] for split in HUGGINGFACE_DATASETS[eval_task][1]}
                    else:
                        self.dataset[eval_task] = load_dataset(HUGGINGFACE_DATASETS[eval_task][0])[HUGGINGFACE_DATASETS[eval_task][1]]
                    
            elif eval_task == 'paws_qqp':
                self.dataset[eval_task] = pd.read_csv('data/paws_qqp/output/dev_and_test.tsv', sep='\t')
            elif eval_task == 'beir':
                print("beir load by itself")
                self.dataset[eval_task] = "BEIR Benchmark"
            elif eval_task in PICKLE_DATASETS:
                with open(f'data/eval/{eval_task}.pkl', 'rb') as f:
                    self.dataset[eval_task] = pickle.load(f)
            elif 'wmt' in eval_task:
                self.dataset[eval_task] = []
                with open(f'data/eval/{eval_task}_eval.jsonl', 'r', encoding='utf8') as f:
                    for example in f:
                        self.dataset[eval_task].append(json.loads(example))
            elif 'true' == eval_task:
                for each_true_sub in os.listdir('data/eval/true'):
                    if 'qags' in each_true_sub:
                        each_true_sub_name = 'true_' + '_'.join(each_true_sub.split('_')[:2])
                    else:
                        each_true_sub_name = 'true_' + '_'.join(each_true_sub.split('_')[:1])

                    self.dataset[each_true_sub_name] = pd.read_csv(os.path.join('data/eval/true', each_true_sub))
            elif 'summac' == eval_task:
                from summac.benchmark import SummaCBenchmark
                self.summac_validation_set = dict()
                summac_benchmark = SummaCBenchmark(benchmark_folder="./data/eval/summac/benchmark", cut='test')
                for each in summac_benchmark.datasets:
                    summac_dt_name = each['name']
                    self.dataset['summac_'+summac_dt_name] = each['dataset']

                summac_benchmark_valid = SummaCBenchmark(benchmark_folder="./data/eval/summac/benchmark", cut='val')
                for each in summac_benchmark_valid.datasets:
                    summac_dt_name = each['name']
                    self.summac_validation_set['summac_'+summac_dt_name] = each['dataset']
            else:
                f = open(f'data/eval/{eval_task}.json')
                self.dataset[eval_task] = json.load(f)
                f.close()
    
    def print_result_table(self, table):
        self.result_tables.append(pd.DataFrame(table).to_markdown())
        self.result_dicts.append(table)
        print(self.result_tables[-1])
    
    def print_all_tables(self):
        print("\n All Evaluation Results:")
        for each in self.result_tables:
            print(each)
            print('='*100)
    
    def save_all_tables(self):
        with open(f'exp_results/{self.result_save_name}.pkl', 'wb') as f:
            pickle.dump(self.result_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)

    def evaluate(self):
        for each_task in self.dataset:
            eval(f'self.evaluate_{each_task}()')
        
        if self.should_save_all_tables:
            self.save_all_tables()

    def get_accuracy(self, true_score, pred_score):
        return [accuracy_score(true_score, [m>0.5 for m in pred_score])]
    
    def get_balanced_accuracy(self, true_score, pred_score, thres=0.5):
        return [balanced_accuracy_score(true_score, [m>thres for m in pred_score])]

    def get_f1(self, true_score, pred_score):
        return [f1_score(true_score, [m>0.5 for m in pred_score])]
    
    def get_3label_f1(self, true_score, pred_score):
        return [f1_score(true_score, pred_score, average='micro')]

    def get_pearson(self, true_score, pred_score):
        return pearsonr(pred_score, true_score)
        
    def get_kendalltau(self, true_score, pred_score):
        return kendalltau(pred_score, true_score)

    def get_spearman(self, true_score, pred_score):
        return spearmanr(pred_score, true_score)
    
    def get_matthews_corr(self, true_score, pred_score):
        return [matthews_corrcoef(true_score, [s>0.5 for s in pred_score])]
    
    
    def clean_text(self, context, claims):
        from nltk.tokenize import sent_tokenize

        if not self.clean_data:
            return claims
        
        word_cases = {token.lower():token for token in context.strip().split()}
        
        def clean(text):
            text = ' '.join(word_cases.get(token.lower(), token) for token in text.strip().split())
            text = text.replace('“', '"').replace('”', '"').replace('’', '\'').replace('‘', '\'').replace('`', '\'').replace('-lrb-', '(').replace('-rrb-', ')')
            text= ' '.join(each.strip()[0].capitalize()+each.strip()[1:] for each in sent_tokenize(text))
            return text
        
        if isinstance(claims, str):
            return clean(claims)
        
        return [clean(text) for text in claims]
        

    def evaluate_newsroom(self):
        true_score = []
        true_score_rel = []
        true_score_binary = []
        sent1 = []
        sent2 = []

        for sample in self.dataset['newsroom'].values():
            summaries, informativeness, relevance = zip(*(
                (s['sys_summ'], s['scores']['informativeness'], s['scores']['relevance'])
                 for s in sample['sys_summs'].values()
            ))
            cleaned_summaries = self.clean_text(sample['src'], summaries)
            for summary, inf_score, rel_score in zip(cleaned_summaries, informativeness, relevance):
                sent1.append(sample['src'])
                sent2.append(summary)
                true_score.append(inf_score)
                true_score_rel.append(rel_score)
                true_score_binary.append(int(inf_score >= 4))

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['newsroom']].tolist()

        self.print_result_table({
            'Dataset_name': 'newsroom',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score_binary, pred_score),
            'Pearson_rel': self.get_pearson(true_score_rel, pred_score),
            'Spearman_rel': self.get_spearman(true_score_rel, pred_score),
            'Kendall_rel': self.get_kendalltau(true_score_rel, pred_score),
        })    

    def evaluate_rank19(self):
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
        true_score = []
        sent1 = []
        sent2 = []
        
        for example in self.dataset['rank19']:
            for example_summs in self.dataset['rank19'][example]['sys_summs']:
                sent1.append(self.dataset['rank19'][example]['src'])
                sent2.append(self.dataset['rank19'][example]['sys_summs'][example_summs]['sys_summ'])
                true_score.append(self.dataset['rank19'][example]['sys_summs'][example_summs]['scores']['fact'])

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['rank19']].tolist()
        pred_score_bin = []
        assert len(pred_score) % 2 == 0
        for i, pair in enumerate(chunks(pred_score, 2)):
            pred_score_bin.extend([0, 1] if pair[1] > pair[0] else [1, 0])

        self.print_result_table({
            'Dataset_name': 'rank19',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score_bin)[0],
            'AUC': roc_auc_score(true_score, pred_score_bin)
        })    

    def evaluate_bagel(self):
        true_score = []
        true_score_binary = []
        sent1 = []
        sent2 = []
        pred_score = []

        for example in self.dataset['bagel']:
            sent1.append(' '.join(self.dataset['bagel'][example]['ref_summs']))
            sent2.append(self.dataset['bagel'][example]['sys_summ'])
            true_score.append(self.dataset['bagel'][example]['scores']['informativeness'])
        
            if(self.dataset['bagel'][example]['scores']['informativeness'] >= 4.0):
                true_score_binary.append(1)
            else:
                true_score_binary.append(0)
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['bagel']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'bagel',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score_binary, pred_score)
        })   

    def evaluate_sfhot(self):
        true_score = []
        sent1 = []
        sent2 = []
        pred_score = []

        for example in self.dataset['sfhot']:
            for ref in self.dataset['sfhot'][example]['ref_summs']:
                sent1.append(self.dataset['sfhot'][example]['sys_summ'])
                sent2.append(ref)
            pred_score.append(max(self.align_func(sent1, sent2)[ALL_TASKS['sfhot']].tolist()))
            sent1 = []
            sent2 = []
            if(self.dataset['sfhot'][example]['scores']['quality'] >= 4.0):
                true_score.append(1)
            else:
                true_score.append(0)

        self.print_result_table({
            'Dataset_name': 'sfhot',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })  

    def evaluate_sfres(self):
        true_score = []
        sent1 = []
        sent2 = []
        pred_score = []

        for example in self.dataset['sfres']:
            for ref in self.dataset['sfres'][example]['ref_summs']:
                sent1.append(self.dataset['sfres'][example]['sys_summ'])
                sent2.append(ref)
            pred_score.append(max(self.align_func(sent1, sent2)[ALL_TASKS['sfres']].tolist()))
            sent1 = []
            sent2 = []
            if(self.dataset['sfres'][example]['scores']['quality'] >= 4.0):
                true_score.append(1)
            else:
                true_score.append(0)

        self.print_result_table({
            'Dataset_name': 'sfres',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        }) 
        
        
    def evaluate_stsb(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['stsb']:
            sent1.append(example['sentence1'])
            sent2.append(example['sentence2'])
            true_score.append(example['label'])

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['stsb']].tolist()

        self.print_result_table({
            'Dataset_name': 'stsb',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score)
        })

    def evaluate_sick(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['sick']:
            sent1.append(example['sentence_A'])
            sent2.append(example['sentence_B'])
            true_score.append(example['relatedness_score'])

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['sick']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'sick-r',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score)
        })

    def evaluate_race_m(self):
        true_score = []
        article = []
        qa = []

        for example in self.dataset['race_m']:
            for i, option in enumerate(example['options']):
                article.append(example['article'])
                qa.append(example['question']+" "+option+" " if "_" not in example['question'] else ' '.join(example['question'].replace("_", " "+option+" ").split()))
                if i == ord(example['answer'])-65:
                    true_score.append(i) # 0,1,2,3

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['race_m']].tolist()
        for a, b, c, d in zip(*[iter(pred_score_temp)]*4):
            arr = [0]*4
            pred_score.append(np.argmax([a,b,c,d]))
            
        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)

        self.print_result_table({
            'Dataset_name': 'race-m',
            'Accuracy': [acc],
        })

    def evaluate_race_h(self):
        true_score = []
        article = []
        qa = []
 
        for example in self.dataset['race_h']:
            for i, option in enumerate(example['options']):
                article.append(example['article'])
                qa.append(example['question']+" "+option+" " if "_" not in example['question'] else ' '.join(example['question'].replace("_", " "+option+" ").split()))
                if i == ord(example['answer'])-65:
                    true_score.append(i) # 0,1,2,3

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['race_h']].tolist()
        for a, b, c, d in zip(*[iter(pred_score_temp)]*4):
            pred_score.append(np.argmax([a,b,c,d]))

        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)

        self.print_result_table({
            'Dataset_name': 'race-h',
            'Accuracy': [acc]
        })

    # How to combine passage, question, and single answer for boolq
    def evaluate_boolq(self):
        true_score = []
        article = []
        qa = []
        for example in self.dataset['boolq']:
            for i in range(2):
                article.append(example['passage'])
                if i == 0:
                    qa.append(example['question']+" "+"No.") # 0
                else:
                    qa.append(example['question']+" "+"Yes.") # 1
            true_score.append(int(example['answer']))

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['boolq']].tolist()
        for a, b in zip(*[iter(pred_score_temp)]*2):
            pred_score.append(np.argmax([a,b]))

        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)
        self.print_result_table({
            'Dataset_name': 'boolq',
            'Accuracy': [acc]
        })

    def evaluate_anli_1(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['anli_1']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])
            true_score.append(example['label'] if example['label']!=-1 else 1)

        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['anli_1']], dim=-1).tolist()
        
        self.print_result_table({
            'Dataset_name': 'anli-1',
            'Accuracy': [accuracy_score(true_score, pred_score)]
        })

    def evaluate_anli_2(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['anli_2']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])
            true_score.append(example['label'] if example['label']!=-1 else 1)

        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['anli_2']], dim=-1).tolist()
        
        self.print_result_table({
            'Dataset_name': 'anli-2',
            'Accuracy': [accuracy_score(true_score, pred_score)]
        })

    def evaluate_anli_3(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['anli_3']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])
            true_score.append(example['label'] if example['label']!=-1 else 1)

        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['anli_3']], dim=-1).tolist()
        
        self.print_result_table({
            'Dataset_name': 'anli-3',
            'Accuracy': [accuracy_score(true_score, pred_score)]
        })

    def evaluate_nli_fever(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['nli_fever']:
            sent1.append(example['hypothesis']) # the original dataset flipped
            sent2.append(example['premise'])
            true_score.append(example['label'] if example['label']!=-1 else 1)

        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['nli_fever']], dim=-1).tolist()
        
        self.print_result_table({
            'Dataset_name': 'nli_fever',
            'Accuracy': [accuracy_score(true_score, pred_score)]
        })

    def evaluate_snli(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['snli']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])
            true_score.append(example['label'] if example['label']!=-1 else 1)

        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['snli']], dim=-1).tolist()
        
        self.print_result_table({
            'Dataset_name': 'snli',
            'Accuracy': [accuracy_score(true_score, pred_score)]
        })
    
    def evaluate_axb(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['axb']:
            sent1.append(example['sentence1'])
            sent2.append(example['sentence2'])

            true_score.append(1 if example['label']==0 else 0)

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['axb']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'axb',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)],
            'Matthews': self.get_matthews_corr(true_score, pred_score)
        })

    def evaluate_axg(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['axg']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])

            true_score.append(1 if example['label']==0 else 0)

        pred_score = self.align_func(sent1, sent2)[2][:,0].tolist()
        
        self.print_result_table({
            'Dataset_name': 'axg',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)],
        })

    def evaluate_cb(self):
        true_score = []
        sent1 = []
        sent2 = []

        for example in self.dataset['cb']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])

            if example['label'] == 0:
                label = 0
            elif example['label'] == 1:
                label = 2
            elif example['label'] == 2:
                label = 1

            true_score.append(label)

        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['cb']], dim=-1).tolist()
        
        self.print_result_table({
            'Dataset_name': 'cb',
            'Accuracy': [accuracy_score(true_score, pred_score)],
        })

    def evaluate_rte(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['rte']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])

            true_score.append(1 if example['label']==0 else 0)

        pred_score = self.align_func(sent1, sent2)[2][:,0].tolist()
        
        self.print_result_table({
            'Dataset_name': 'rte',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)],
        })

    def evaluate_wnli(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['wnli']:
            sent1.append(example['text1'])
            sent2.append(example['text2'])

            true_score.append(example['label'])

        pred_score = self.align_func(sent1, sent2)[2][:,0].tolist()
        
        self.print_result_table({
            'Dataset_name': 'wnli',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)],
        })

    def evaluate_doc_nli(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['doc_nli']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])

            true_score.append(1 if example['label'] == 'entailment' else 0)

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['doc_nli']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'doc_nli',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)],
        })

    def evaluate_qnli(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['qnli']:
            sent1.append(example['sentence'])
            sent2.append(example['question'])

            true_score.append(1 if example['label'] == 0 else 0)

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['qnli']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'qnli',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)],
        })

    def evaluate_dream(self):
        true_score = []
        article = []
        qa = []

        for example in self.dataset['dream']:
            for i, option in enumerate(example['choice']):
                article.append(' '.join(example['dialogue']))
                qa.append(example['question']+" "+option+" ")
                if option == example['answer']:
                    true_score.append(i) # 0,1,2,3

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['dream']].tolist()
        for a, b, c in zip(*[iter(pred_score_temp)]*3):
            arr = [0]*3
            pred_score.append(np.argmax([a,b,c]))
            
        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)

        self.print_result_table({
            'Dataset_name': 'dream',
            'Accuracy': [acc],
        })

    def evaluate_quartz(self):
        true_score = []
        article = []
        qa = []

        for example in self.dataset['quartz']:
            for i, option in enumerate(example['choices']['text']):
                article.append(example['para'])
                qa.append(example['question']+" "+option+" ")
                if i == ord(example['answerKey'])-65:
                    true_score.append(i) # 0,1,2,3

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['quartz']].tolist()
        for a, b in zip(*[iter(pred_score_temp)]*2):
            arr = [0]*2
            pred_score.append(np.argmax([a,b]))
            
        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)

        self.print_result_table({
            'Dataset_name': 'quartz',
            'Accuracy': [acc],
        })
    def evaluate_eraser_multi_rc(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['eraser_multi_rc']:
            sent1.append(example['passage'])
            sent2.append(example['query_and_answer'].replace("|", ""))
            true_score.append(example['label'])
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['eraser_multi_rc']].tolist()

        self.print_result_table({
            'Dataset_name': 'eraser_multi_rc',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)]
        })

    def evaluate_quail(self):
        true_score = []
        article = []
        qa = []

        for example in self.dataset['quail']:
            for i, option in enumerate(example['answers']):
                article.append(example['context'])
                qa.append(example['question']+" "+option+" ")
                if i == example['correct_answer_id']:
                    true_score.append(i) # 0,1,2,3

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['quail']].tolist()
        for a, b, c, d in zip(*[iter(pred_score_temp)]*4):
            arr = [0]*4
            pred_score.append(np.argmax([a,b,c,d]))
            
        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)

        self.print_result_table({
            'Dataset_name': 'quail',
            'Accuracy': [acc],
        })

    def evaluate_sciq(self):
        true_score = []
        article = []
        qa = []

        for example in self.dataset['sciq']:
            options = [example['correct_answer'], example['distractor1'], example['distractor2'], example['distractor3']]
            for i, option in enumerate(options):
                article.append(example['support'])
                qa.append(example['question']+" "+option+" ")
                if i == 0:
                    true_score.append(i) # 0,1,2,3, always 0

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['sciq']].tolist()
        for a, b, c, d in zip(*[iter(pred_score_temp)]*4):
            arr = [0]*4
            pred_score.append(np.argmax([a,b,c,d]))
            
        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)

        self.print_result_table({
            'Dataset_name': 'sciq',
            'Accuracy': [acc],
        })

    def evaluate_gap(self):
        true_score = []
        article = []
        qa = []

        for example in self.dataset['gap']:
            options = [example['Text'][:example['Pronoun-offset']]+example['A']+example['Text'][(example['Pronoun-offset']+len(example['Pronoun'])):],
                       example['Text'][:example['Pronoun-offset']]+example['B']+example['Text'][(example['Pronoun-offset']+len(example['Pronoun'])):]]
            for i, option in enumerate(options):
                article.append(example['Text'])
                qa.append(option)
                
            true_score.append(1 if example['B-coref'] else 0) # 0,1,2,3, always 0

        pred_score = []
        pred_score_temp = self.align_func(article, qa)[ALL_TASKS['gap']].tolist()
        for a, b in zip(*[iter(pred_score_temp)]*2):
            pred_score.append(np.argmax([a,b]))
            
        assert len(pred_score) == len(true_score)
        acc = [int(p==t) for p, t in zip(pred_score, true_score)]
        acc = sum(acc) / len(acc)

        self.print_result_table({
            'Dataset_name': 'gap',
            'Accuracy': [acc],
        })
    
    # How to group fact checking
    def evaluate_vitaminc(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['vitaminc']:
            sent1.append(example['evidence'])
            sent2.append(example['claim'])
            if example['label'] == 'SUPPORTS':
                true_score.append(0)
            elif example['label'] == 'REFUTES':
                true_score.append(2)
            else:
                true_score.append(1)

        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['vitaminc']], dim=-1).tolist()
        
        self.print_result_table({
            'Dataset_name': 'vitaminc',
            'F1': self.get_3label_f1(true_score, pred_score),
            'Accuracy': [accuracy_score(true_score, pred_score)],
        })

    def evaluate_mrpc(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['mrpc']:
            sent1.append(example['sentence1'])
            sent2.append(example['sentence2'])
            true_score.append(example['label'])
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['mrpc']].tolist()

        self.print_result_table({
            'Dataset_name': 'mrpc',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)]
        })

    def evaluate_paws(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['paws']:
            sent1.append(example['sentence1'])
            sent2.append(example['sentence2'])
            true_score.append(example['label'])
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['paws']].tolist()

        self.print_result_table({
            'Dataset_name': 'paws',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)]
        })

    def evaluate_mnli_matched(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['mnli_matched']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])
            true_score.append(example['label'] if example['label']!=-1 else 1)
        
        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['mnli_matched']], dim=-1).tolist()

        self.print_result_table({
            'Dataset_name': 'mnli_matched',
            'Accuracy': [accuracy_score(true_score, pred_score)]
        })

    def evaluate_mnli_mismatched(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['mnli_mismatched']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])
            true_score.append(example['label'] if example['label']!=-1 else 1)
        
        pred_score = torch.argmax(self.align_func(sent1, sent2)[ALL_TASKS['mnli_mismatched']], dim=-1).tolist()

        self.print_result_table({
            'Dataset_name': 'mnli_mismatched',
            'Accuracy': [accuracy_score(true_score, pred_score)]
        })

    def evaluate_sem_eval(self):
        print('Reached here')
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['sem_eval']:
            sent1.append(example['premise'])
            sent2.append(example['hypothesis'])
            if example['entailment_judgment'] == 1:
                true_score.append(1)
            else:
                true_score.append(0)
            
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['sem_eval']].tolist()

        self.print_result_table({
            'Dataset_name': 'sem_eval',
            'Accuracy': self.get_accuracy(true_score, pred_score)
        })
    
    def evaluate_summeval(self):
        true_score = []
        true_score_rel = []
        true_score_binary = []
        pred_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['summeval']:
            cleaned_summary = self.clean_text(example['document'], example['summary'])
            sent1.append(example['document'])
            sent2.append(cleaned_summary)
            true_score.append(example['consistency'])
            true_score_rel.append(example['relevance'])
            true_score_binary.append(1 if example['consistency'] == 5.0 else 0)
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['summeval']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'summeval',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score_binary, pred_score),
            'Pearson_rel': self.get_pearson(true_score_rel, pred_score),
            'Spearman_rel': self.get_spearman(true_score_rel, pred_score),
            'Kendall_rel': self.get_kendalltau(true_score_rel, pred_score),
        })

    def evaluate_qags_xsum(self):
        true_score = []
        pred_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['qags_xsum']:
            sent1.append(example['document'])
            sent2.append(example['summary'])
            true_score.append(example['consistency'])
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['qags_xsum']].tolist()
             
        self.print_result_table({
            'Dataset_name': 'qags_xsum',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })

    def evaluate_qags_cnndm(self):
        true_score = []
        pred_score = []
        sent1 = []
        sent2 = []
        true_score_binary = []
        for example in self.dataset['qags_cnndm']:
            sent1.append(example['document'])
            sent2.append(example['summary'])
            true_score.append(example['consistency'])
            true_score_binary.append(1 if example['consistency'] == 1.0 else 0)
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['qags_cnndm']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'qags_cnndm',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score_binary, pred_score)
        })

    def evaluate_frank(self):
        from spacy.lang.en import English
        nlp = English()
        nlp.add_pipe("sentencizer")
        for d in self.dataset['frank']:
            if d['dataset'] == 'cnndm':
                continue
            d['document'] = ' '.join([each.text for each in nlp(d['document']).sents])

        true_score_xsum = []
        true_score_cnndm = []
        pred_score_xsum = []
        pred_score_cnndm = []
        sent1_xsum = []
        sent1_cnndm = []
        sent2_xsum = []
        sent2_cnndm = []
        true_score_binary_cnndm = []
        true_score_binary_xsum = []
        for example in self.dataset['frank']:
            if example['dataset'] == 'cnndm': 
                sent1_cnndm.append(example['document'])
                sent2_cnndm.append(self.clean_text(example['document'], example['summary']))
                true_score_cnndm.append(example['score'])
                true_score_binary_cnndm.append(1 if example['score'] == 1.0 else 0)
            elif example['dataset'] == 'xsum': 
                sent1_xsum.append(example['document'])
                sent2_xsum.append(self.clean_text(example['document'], example['summary']))
                true_score_xsum.append(example['score'])
                true_score_binary_xsum.append(1 if example['score'] == 1.0 else 0)
        
        pred_score_xsum = self.align_func(sent1_xsum, sent2_xsum)[ALL_TASKS['frank']].tolist() #
        pred_score_cnndm = self.align_func(sent1_cnndm, sent2_cnndm)[ALL_TASKS['frank']].tolist() #
        
        self.print_result_table({
            'Dataset_name': 'frank-xsum',
            'Pearson': self.get_pearson(true_score_xsum, pred_score_xsum),
            'Spearman': self.get_spearman(true_score_xsum, pred_score_xsum),
            'Kendall': self.get_kendalltau(true_score_xsum, pred_score_xsum),
            'AUC': roc_auc_score(true_score_binary_xsum, pred_score_xsum)
        })

        self.print_result_table({
            'Dataset_name': 'frank-cnndm',
            'Pearson': self.get_pearson(true_score_cnndm, pred_score_cnndm),
            'Spearman': self.get_spearman(true_score_cnndm, pred_score_cnndm),
            'Kendall': self.get_kendalltau(true_score_cnndm, pred_score_cnndm),
            'AUC': roc_auc_score(true_score_binary_cnndm, pred_score_cnndm)
        })

        self.print_result_table({
            'Dataset_name': 'frank-all',
            'Pearson': self.get_pearson(true_score_xsum+true_score_cnndm, pred_score_xsum+pred_score_cnndm),
            'Spearman': self.get_spearman(true_score_xsum+true_score_cnndm, pred_score_xsum+pred_score_cnndm),
            'Kendall': self.get_kendalltau(true_score_xsum+true_score_cnndm, pred_score_xsum+pred_score_cnndm),
            'AUC': roc_auc_score(true_score_binary_xsum+true_score_binary_cnndm, pred_score_xsum+pred_score_cnndm)
        })

    def evaluate_xsumfaith(self):
        dataset_name = 'xsumfaith'

        true_score = []
        pred_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset[dataset_name]:
            sent1.append(example['document'])
            sent2.append(self.clean_text(example['document'], example['claim']))
            true_score.append(example['label'])
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS[dataset_name]].tolist()
             
        self.print_result_table({
            'Dataset_name': dataset_name,
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
        })

    def evaluate_samsum(self):
        dataset_name = 'samsum'

        label_mapping = {
            'factual': 1,
            'factually incorrect': 0,
            'too incoherent': 0
        }
        import string
        printable = set(string.printable)
        

        true_score = []
        pred_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset[dataset_name]:
            cleaned_doc = ''.join(filter(lambda x: x in printable, example['article']))
            sent1.append(cleaned_doc)
            sent2.append(example['summary'])
            true_score.append(label_mapping[example['label']])
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS[dataset_name]].tolist()
             
        self.print_result_table({
            'Dataset_name': dataset_name,
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })
    def evaluate_yelp(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['yelp']:
            sent1.append(example['input_sent'])
            sent2.append(example['output_sent'])
            true_score.append(example['preservation'])

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['yelp']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'yelp',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score)
        })

    def evaluate_persona_chat(self):
        true_score = []
        pred_score = []
        premise = []
        hypothesis = []
        for example in self.dataset['persona_chat']:
            premise.append(example['dialog_history']+example['fact'])
            hypothesis.append(example['response'])
            true_score.append(example['engaging'])
        pred_score = self.align_func(premise, hypothesis)[ALL_TASKS['persona_chat']].tolist()

        self.print_result_table({
            'Dataset_name': 'persona_chat_eng',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score)
        })

        true_score = []
        pred_score = []
        premise = []
        hypothesis = []
        for example in self.dataset['persona_chat']:
            premise.append(example['fact'])
            hypothesis.append(example['response'])
            true_score.append(example['uses_knowledge'])
        pred_score = self.align_func(premise, hypothesis)[ALL_TASKS['persona_chat']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'persona_chat_grd',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score)
        })

    def evaluate_topical_chat(self):
        true_score = []
        pred_score = []
        premise = []
        hypothesis = []
        for example in self.dataset['topical_chat']:
            premise.append(example['dialog_history']+example['fact'])
            hypothesis.append(example['response'])
            true_score.append(example['engaging'])
        pred_score = self.align_func(premise, hypothesis)[ALL_TASKS['topical_chat']].tolist()
      
        self.print_result_table({
            'Dataset_name': 'topical_chat_eng',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score)
        })

        true_score = []
        pred_score = []
        premise = []
        hypothesis = []
        for example in self.dataset['topical_chat']:
            premise.append(example['fact'])
            hypothesis.append(example['response'])
            true_score.append(example['uses_knowledge'])
        pred_score = self.align_func(premise, hypothesis)[ALL_TASKS['topical_chat']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'topical_chat_grd',
            'Pearson': self.get_pearson(true_score, pred_score),
            'Spearman': self.get_spearman(true_score, pred_score),
            'Kendall': self.get_kendalltau(true_score, pred_score)
        })

    def evaluate_paws_qqp(self):
        sent1 = []
        sent2 = []
        true_score = []
        for i in range(self.dataset['paws_qqp']['label'].size):
            sent1.append(self.dataset['paws_qqp']['sentence1'][i][2:-1])
            sent2.append(self.dataset['paws_qqp']['sentence2'][i][2:-1])
            true_score.append(self.dataset['paws_qqp']['label'][i])
        
        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['paws_qqp']].tolist()
        roc_auc = roc_auc_score(true_score, pred_score)
        
        self.print_result_table({
            'Dataset_name': 'paws_qqp',
            'F1': self.get_f1(true_score, pred_score), 
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc]
        })
    
    def evaluate_qqp(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['qqp']:
            sent1.append(example['question1'])
            sent2.append(example['question2'])
            true_score.append(example['label'])

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['qqp']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'qqp',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc_score(true_score, pred_score)]
        })

    def evaluate_wmt17(self):
        lang_pair = list(set([each['lang'] for each in self.dataset['wmt17']]))

        for each_lang_pair in lang_pair:
            true_score = []
            premise = []
            hypothesis = []
            for example in self.dataset['wmt17']:
                if example['lang'] != each_lang_pair:
                    continue
                premise.append(example['reference'])
                hypothesis.append(example['candidate'])
                true_score.append(example['score'])
            pred_score = self.align_func(premise, hypothesis)[ALL_TASKS['wmt17']].tolist()
            
            self.print_result_table({
                'Dataset_name': f'wmt17-{each_lang_pair}',
                'Pearson': self.get_pearson(true_score, pred_score),
                'Spearman': self.get_spearman(true_score, pred_score),
                'Kendall': self.get_kendalltau(true_score, pred_score)
            })
    
    def evaluate_wmt18(self):
        lang_pair = list(set([each['lang'] for each in self.dataset['wmt18']]))

        for each_lang_pair in lang_pair:
            true_score = []
            premise = []
            hypothesis = []
            for example in self.dataset['wmt18']:
                if example['lang'] != each_lang_pair:
                    continue
                premise.append(example['reference'])
                hypothesis.append(example['candidate'])
                true_score.append(example['score'])
            pred_score = self.align_func(premise, hypothesis)[ALL_TASKS['wmt18']].tolist()
            
            self.print_result_table({
                'Dataset_name': f'wmt18-{each_lang_pair}',
                'Pearson': self.get_pearson(true_score, pred_score),
                'Spearman': self.get_spearman(true_score, pred_score),
                'Kendall': self.get_kendalltau(true_score, pred_score)
            })
    def evaluate_wmt19(self):
        lang_pair = list(set([each['lang'] for each in self.dataset['wmt19']]))

        for each_lang_pair in lang_pair:
            true_score = []
            premise = []
            hypothesis = []
            for example in self.dataset['wmt19']:
                if example['lang'] != each_lang_pair:
                    continue
                premise.append(example['reference'])
                hypothesis.append(example['candidate'])
                true_score.append(example['score'])
            pred_score = self.align_func(premise, hypothesis)[ALL_TASKS['wmt19']].tolist()
            
            self.print_result_table({
                'Dataset_name': f'wmt19-{each_lang_pair}',
                'Pearson': self.get_pearson(true_score, pred_score),
                'Spearman': self.get_spearman(true_score, pred_score),
                'Kendall': self.get_kendalltau(true_score, pred_score)
            })

    def evaluate_sst2(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['sst2']:
            sent1.append(example['text'])
            sent2.append('It was great.')
            true_score.append(int(example['label_text'] == 'positive'))

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['sst2']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'sst2',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })

    def evaluate_cr(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['cr']:
            sent1.append(example['text'])
            sent2.append('It was great.')
            true_score.append(int(example['label_text'] == 'positive'))

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['cr']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'cr',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })

    def evaluate_subj(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['subj']:
            sent1.append(example['text'])
            sent2.append('It was objective.')
            true_score.append(int(example['label_text'] == 'objective'))

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['subj']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'subj',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })

    def evaluate_imdb(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['imdb']:
            sent1.append(example['text'])
            sent2.append('It was great.')
            true_score.append(int(example['label_text'] == 'positive'))

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['imdb']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'imdb',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })

    def evaluate_imdb_knn(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['imdb']:
            sent1.append(example['text'])
            sent2.append('It was great.')
            true_score.append(int(example['label_text'] == 'positive'))

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['imdb']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'imdb',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })

    def evaluate_cola(self):
        true_score = []
        sent1 = []
        sent2 = []
        for example in self.dataset['cola']:
            sent1.append(example['sentence'])
            sent2.append('It was correct.')
            true_score.append(example['label'])

        pred_score = self.align_func(sent1, sent2)[ALL_TASKS['cola']].tolist()
        
        self.print_result_table({
            'Dataset_name': 'cola',
            'F1': self.get_f1(true_score, pred_score),
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': roc_auc_score(true_score, pred_score)
        })

    def evaluate_yelp_efl(self):
        sent = []
        label = []
        for example in self.dataset['yelp_efl']:
            sent.append(example['text'])
            label.append(example['label'])
        templates = [
            'It was terrible.',
            'It was bad.',
            'It was ok.',
            'It was good.',
            'It was great.',
        ]
        template_lists = [[template] * len(sent) for template in templates]
        predictions = [
            self.align_func(sent, template_list)[ALL_TASKS['yelp_efl']]
            for template_list in template_lists
        ]

        pred_label = torch.argmax(torch.stack(predictions), dim=0).tolist()
        
        self.print_result_table({
            'Dataset_name': 'yelp_efl',
            'Accuracy': [accuracy_score(label, pred_label)]
        })

    def evaluate_ag_news(self):
        sent = []
        label = []
        for example in self.dataset['ag_news']:
            sent.append(example['text'])
            label.append(example['label'])
        templates = [
            'It is world news.',
            'It is sports news.',
            'It is business news.',
            'It is science news.',
        ]
        template_lists = [[template] * len(sent) for template in templates]
        predictions = [
            self.align_func(sent, template_list)[ALL_TASKS['ag_news']]
            for template_list in template_lists
        ]

        pred_label = torch.argmax(torch.stack(predictions), dim=0).tolist()
        
        self.print_result_table({
            'Dataset_name': 'ag_news',
            'Accuracy': [accuracy_score(label, pred_label)]
        })

    def evaluate_trec(self):
        sent = []
        label = []
        for example in self.dataset['trec']:
            sent.append(example['text'])
            label.append(example['label_coarse'])
        templates = [
            'It is description.',
            'It is entity.',
            'It is expression.',
            'It is human.',
            'It is number.',
            'It is location.',
        ]
        template_lists = [[template] * len(sent) for template in templates]
        predictions = [
            self.align_func(sent, template_list)[ALL_TASKS['trec']]
            for template_list in template_lists
        ]

        pred_label = torch.argmax(torch.stack(predictions), dim=0).tolist()
        
        self.print_result_table({
            'Dataset_name': 'trec',
            'Accuracy': [accuracy_score(label, pred_label)]
        })

    def true_task_helper(self, dataset_name):
        sent1 = []
        sent2 = []
        true_score = []
        for i in range(len(self.dataset[dataset_name])):
            context = self.dataset[dataset_name].iloc[i]['grounding']
            claim = self.dataset[dataset_name].iloc[i]['generated_text']
            sent1.append(context)
            sent2.append(self.clean_text(context, claim))
            true_score.append(self.dataset[dataset_name].iloc[i]['label'])
        
        pred_score = self.align_func(sent1, sent2)[1].tolist()
        roc_auc = roc_auc_score(true_score, pred_score)
        
        self.print_result_table({
            'Dataset_name': dataset_name,
            'F1': self.get_f1(true_score, pred_score), 
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'AUC': [roc_auc]
        })

    def evaluate_true_begin(self):
        dataset_name = 'true_begin'
        self.true_task_helper(dataset_name)

        
    def evaluate_true_dialfact(self):
        dataset_name = 'true_dialfact'
        self.true_task_helper(dataset_name)

    def evaluate_true_fever(self):
        dataset_name = 'true_fever'
        self.true_task_helper(dataset_name)

    def evaluate_true_frank(self):
        dataset_name = 'true_frank'
        self.true_task_helper(dataset_name)

    def evaluate_true_mnbm(self):
        dataset_name = 'true_mnbm'
        self.true_task_helper(dataset_name)

    def evaluate_true_paws(self):
        dataset_name = 'true_paws'
        self.true_task_helper(dataset_name)
        
    def evaluate_true_q2(self):
        dataset_name = 'true_q2'
        self.true_task_helper(dataset_name)
        
    def evaluate_true_qags_cnndm(self):
        dataset_name = 'true_qags_cnndm'
        self.true_task_helper(dataset_name)

    def evaluate_true_qags_xsum(self):
        dataset_name = 'true_qags_xsum'
        self.true_task_helper(dataset_name)

    def evaluate_true_summeval(self):
        dataset_name = 'true_summeval'
        self.true_task_helper(dataset_name)

    def evaluate_true_vitc(self):
        dataset_name = 'true_vitc'
        self.true_task_helper(dataset_name)

    def get_summac_thres(self, dataset_name):
        sent1 = []
        sent2 = []
        true_score = []
        for example in self.summac_validation_set[dataset_name]:
            sent1.append(example['document'])
            sent2.append(self.clean_text(example['document'], example['claim'])) #
            true_score.append(example['label'])
        
        pred_score = self.align_func(sent1, sent2)[1].tolist()

        thres_result = []
        for i in range(1001):
            thres = i / 1000
            thres_result.append((thres, balanced_accuracy_score(true_score, [p>thres for p in pred_score])))
        
        best_thres = sorted(thres_result, key=lambda x: x[1], reverse=True)[0]
        print(f"best thres for {dataset_name} is {best_thres[0]} @ {best_thres[1]}")

        return best_thres[0]
    
    def summac_task_helper(self, dataset_name):
        sent1 = []
        sent2 = []
        true_score = []
        for example in self.dataset[dataset_name]:
            sent1.append(example['document'])
            sent2.append(self.clean_text(example['document'], example['claim']))
            true_score.append(example['label'])
        
        pred_score = self.align_func(sent1, sent2)[1].tolist()
        roc_auc = roc_auc_score(true_score, pred_score)

        balanced_acc_thres = self.get_summac_thres(dataset_name)
        
        self.print_result_table({
            'Dataset_name': dataset_name,
            'F1': self.get_f1(true_score, pred_score), 
            'Accuracy': self.get_accuracy(true_score, pred_score),
            'BalancedAcc': self.get_balanced_accuracy(true_score, pred_score, thres=balanced_acc_thres), 
            'threshold': balanced_acc_thres,
            'AUC': [roc_auc]
        })

    def evaluate_summac_cogensumm(self):
        dataset_name = 'summac_cogensumm'
        self.summac_task_helper(dataset_name)
        
    def evaluate_summac_xsumfaith(self):
        dataset_name = 'summac_xsumfaith'
        self.summac_task_helper(dataset_name)

    def evaluate_summac_polytope(self):
        dataset_name = 'summac_polytope'
        self.summac_task_helper(dataset_name)

    def evaluate_summac_factcc(self):
        dataset_name = 'summac_factcc'
        self.summac_task_helper(dataset_name)

    def evaluate_summac_summeval(self):
        dataset_name = 'summac_summeval'
        self.summac_task_helper(dataset_name)

    def evaluate_summac_frank(self):
        dataset_name = 'summac_frank'
        self.summac_task_helper(dataset_name)

    def evaluate_beir(self):
        from beir import util, LoggingHandler
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval.search.lexical import BM25Search as BM25
        from beir.reranking.models import CrossEncoder
        from beir.reranking import Rerank

        import pathlib, os
        import logging
        import random

        #### Just some code to print debug information to stdout
        logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])
        #### /print debug information to stdout

        #### Download trec-covid.zip dataset and unzip the dataset
        for beir_dataset_name in ['msmarco', 'trec-covid', 'nfcorpus', 'nq', 'hotpotqa', 'fiqa',
                                  'arguana', 'webis-touche2020', 'cqadupstack', 'quora',
                                  'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact']:
        # for beir_dataset_name in ['fever']:
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(beir_dataset_name)
            # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
            out_dir = f"./data/eval/beir/{beir_dataset_name}/"
            data_path = util.download_and_unzip(url, out_dir)

            #### Provide the data path where trec-covid has been downloaded and unzipped to the data loader
            # data folder would contain these files: 
            # (1) trec-covid/corpus.jsonl  (format: jsonlines)
            # (2) trec-covid/queries.jsonl (format: jsonlines)
            # (3) trec-covid/qrels/test.tsv (format: tsv ("\t"))

            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

            #########################################
            #### (1) RETRIEVE Top-100 docs using BM25
            #########################################

            #### Provide parameters for Elasticsearch
            # print(corpus)
            hostname = "localhost" #localhost
            index_name = beir_dataset_name # trec-covid
            initialize = True # False

            model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
            retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,100,1000])

            #### Retrieve dense results (format of results is identical to qrels)
            results = retriever.retrieve(corpus, queries)

            # Rerank top-100 results using the reranker provided
            reranker = Rerank(self.align_func)
            rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

            #### Evaluate your retrieval using NDCG@k, MAP@K ...
            ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

            self.print_result_table({
                'Dataset_name': beir_dataset_name,
                'ndcg': ndcg, 
                'map': _map,
                'recall': recall, 
                'precision': precision
            })
    def evaluate_xxx(self):
        pass

class evaluateMultiCheckpoints:
    def __init__(self, config, device='cuda:0') -> None:
        sample_checkpoint = {
            'backbone': 'roberta-base',
            'task_name': 'align-wo-finetune | align-finetune | roberta-finetune-baseline | nli-wo-finetune | nli-finetune',
            'path': 'some path',
            'result_save_path': 'some path'
        }
        self.config = config ## a dictionary
        self.device = device

        self.tasks = [
                        'summeval', 'qags_xsum', 'qags_cnndm', 'persona_chat', 'topical_chat',
                        'mnli_mismatched', 'mnli_matched', 
                        'sick', 'yelp', 'stsb', 
                        'anli_1','anli_2', 'anli_3', 'snli', 'vitaminc',
                        'mrpc', 'paws', 'sem_eval', 'paws_qqp', 'qqp',
                        'newsroom', 'rank19', 'bagel', 'race_m', 'race_h'
                        ]

    def experimentForSlide1216(self):
        for ckpt in self.config:
            self.evaluateOneCheckpoint(ckpt)
    def evaluateOneCheckpoint(self, ckpt):
        model_name = ckpt['path'].split('/')[-1].split('.ckpt')[0]
        infer = Inferencer(ckpt_path=ckpt['path'],
                        model=ckpt['backbone'], batch_size=32, device=self.device)
        evaluator = Evaluator(eval_tasks=self.tasks, align_func=infer.inference, save_all_tables=True)

        evaluator.result_save_name = f"{ckpt['result_save_path']}{model_name}"
        evaluator.evaluate()