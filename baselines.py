from logging import warning
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import json

class CTCScorer():
    def __init__(self, model_type) -> None:
        self.model_type = model_type
        import nltk
        nltk.download('stopwords')

        from ctc_score import StyleTransferScorer, SummarizationScorer, DialogScorer
        if model_type == 'D-cnndm':
            self.scorer = SummarizationScorer(align='D-cnndm')
        elif model_type =='E-roberta':
            self.scorer = SummarizationScorer(align='E-roberta')
        elif model_type == 'R-cnndm':
            self.scorer = SummarizationScorer(align='R-cnndm')
    def score(self, premise: list, hypo: list):
        assert len(premise) == len(hypo), "Premise and hypothesis should have the same length"
        
        output_scores = []
        for one_pre, one_hypo in tqdm(zip(premise, hypo), total=len(premise), desc="Evaluating by ctc"):
            score_for_this_example = self.scorer.score(doc=one_pre, refs=[], hypo=one_hypo, aspect='consistency')
            if score_for_this_example is not None:
                output_scores.append(score_for_this_example)
            else:
                output_scores.append(1e-8)
        output = None, torch.tensor(output_scores), None

        return output

class SimCSEScorer():
    def __init__(self, model_type, device) -> None:
        self.model_type = model_type
        self.device = device
        from transformers import AutoModel, AutoTokenizer

        # refer to the model list on https://github.com/princeton-nlp/SimCSE for the list of models
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.model = AutoModel.from_pretrained(model_type).to(self.device)
        self.spacy = spacy.load('en_core_web_sm')

        self.batch_size = 64

    def score(self, premise: list, hypo: list):
        assert len(premise) == len(hypo)

        output_scores = []
        premise_sents = []
        premise_index = [0]
        hypo_sents = []
        hypo_index = [0]

        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc="Sentenizing", total=len(premise)):
            premise_sent = sent_tokenize(one_pre) #[each.text for each in self.spacy(one_pre).sents]
            hypo_sent = sent_tokenize(one_hypo) #[each.text for each in self.spacy(one_hypo).sents]
            premise_sents.extend(premise_sent)
            premise_index.append(len(premise_sents))

            hypo_sents.extend(hypo_sent)
            hypo_index.append(len(hypo_sents))

        all_sents = premise_sents + hypo_sents
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(self.chunks(all_sents, self.batch_size), total=int(len(all_sents)/self.batch_size), desc="Evaluating by SimCSE"):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                embeddings.append(self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output)
            embeddings = torch.cat(embeddings)

            assert len(premise_index) == len(hypo_index)
            for i in range(len(premise_index)-1):
                premise_embeddings = embeddings[premise_index[i]: premise_index[i+1]]
                hypo_embeddings = embeddings[len(premise_sents)+hypo_index[i]:len(premise_sents)+hypo_index[i+1]]
                cos_sim = cosine_similarity(premise_embeddings.cpu(), hypo_embeddings.cpu())
                score_p = cos_sim.max(axis=0).mean()
                score_r = cos_sim.max(axis=1).mean()
                score_f = 2 * score_p * score_r / (score_p + score_r)
                output_scores.append(score_f)

        return torch.Tensor(output_scores), torch.Tensor(output_scores), None
    
    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

class BleurtScorer():
    def __init__(self, checkpoint) -> None:
        self.checkpoint = checkpoint

        from bleurt import score
        # BLEURT-20 can also be switched to other checkpoints to improve time
        # No avaliable api to specify cuda number
        self.model = score.BleurtScorer(self.checkpoint)

    def scorer(self, premise:list, hypo: list):
        assert len(premise) == len(hypo)

        output_scores = self.model.score(references=premise, candidates=hypo, batch_size=8)
        output_scores = [s for s in output_scores]
        return torch.Tensor(output_scores), torch.Tensor(output_scores), torch.Tensor(output_scores)

class BertScoreScorer():
    def __init__(self, model_type, metric, device, batch_size) -> None:
        self.model_type = model_type
        self.device = device
        self.metric = metric
        self.batch_size = batch_size

        from bert_score import score
        self.model = score
    
    def scorer(self, premise: list, hypo: list):
        assert len(premise) == len(hypo)

        precision, recall, f1 = self.model(premise, hypo, model_type=self.model_type, lang='en', rescale_with_baseline=True, verbose=True, device=self.device, batch_size=self.batch_size)

        f1 = [f for f in f1]
        precision = [p for p in precision]
        recall = [r for r in recall]

        if self.metric == 'f1':
            return torch.Tensor(f1), torch.Tensor(f1), None
        elif self.metric == 'precision':
            return torch.Tensor(precision), torch.Tensor(precision), None
        elif self.metric == 'recall':
            return torch.Tensor(recall), torch.Tensor(recall), None
        else:
            ValueError("metric type not in f1, precision or recall.")

class BartScoreScorer():
    def __init__(self, checkpoint, device) -> None:
        self.checkpoint = checkpoint
        self.device = device
        import os, sys
        sys.path.append('baselines/BARTScore')
        from bart_score import BARTScorer
        self.model = BARTScorer(device=self.device, checkpoint=self.checkpoint)
    
    def scorer(self, premise: list, hypo: list):
        assert len(premise) == len(hypo)

        output_scores = self.model.score(premise, hypo, batch_size=4)
        normed_score = torch.exp(torch.Tensor(output_scores))
        
        return normed_score, normed_score, normed_score

### Below are baselines in SummaC
### MNLI, NER, FactCC, DAE, FEQA, QuestEval, SummaC-ZS, SummaC-Conv
class MNLIScorer():
    def __init__(self, model="roberta-large-mnli", device='cuda:0', batch_size=32) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.device = device
        self.softmax = nn.Softmax(dim=-1)
        self.batch_size = batch_size

    def scorer(self, premise: list, hypo: list):
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]
        
        batch = self.batch_tokenize(premise, hypo)
        output_score_tri = []

        for mini_batch in tqdm(batch, desc="Evaluating MNLI"):
        # for mini_batch in batch:
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(**mini_batch)
                model_output_tri = model_output.logits
                model_output_tri = self.softmax(model_output_tri).cpu()

            output_score_tri.append(model_output_tri[:,2])

        output_score_tri = torch.cat(output_score_tri)
        
        return output_score_tri, output_score_tri, output_score_tri

    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(hypo), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
            try:
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation='only_first', padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            except:
                warning('text_b too long...')
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            batch.append(mini_batch)

        return batch

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

class NERScorer():
    def __init__(self) -> None:
        import os, sys
        sys.path.append('baselines/summac/summac')
        from model_guardrails import NERInaccuracyPenalty
        self.ner = NERInaccuracyPenalty()
    
    def scorer(self, premise, hypo):
        score_return = self.ner.score(premise, hypo)['scores']
        oppo_score = [float(not each) for each in score_return]
        
        tensor_score = torch.tensor(oppo_score)

        return tensor_score, tensor_score, tensor_score
class UniEvalScorer():
    def __init__(self, task='fact', device='cuda:0') -> None:
        import os, sys
        sys.path.append('baselines/UniEval')
        from metric.evaluator import get_evaluator

        self.evaluator = get_evaluator(task, device=device)
    
    def scorer(self, premise, hypo):
        from utils import convert_to_json
        # Prepare data for pre-trained evaluators
        data = convert_to_json(output_list=hypo, src_list=premise)
        # Initialize evaluator for a specific task
        
        # Get factual consistency scores
        eval_scores = self.evaluator.evaluate(data, print_result=True)
        score_list = [each['consistency'] for each in eval_scores]

        return torch.tensor(score_list), torch.tensor(score_list), torch.tensor(score_list)

class FEQAScorer():
    def __init__(self) -> None:
        import os, sys
        sys.path.append('baselines/feqa')
        import benepar
        import nltk

        benepar.download('benepar_en3')
        nltk.download('stopwords')

        from feqa import FEQA
        self.feqa_model = FEQA(squad_dir=os.path.abspath('baselines/feqa/qa_models/squad1.0'), bart_qa_dir=os.path.abspath('baselines/feqa/bart_qg/checkpoints/'), use_gpu=True)
    
    def scorer(self, premise, hypo):
        eval_score = self.feqa_model.compute_score(premise, hypo, aggregate=False)

        return torch.tensor(eval_score), torch.tensor(eval_score), torch.tensor(eval_score)


class QuestEvalScorer():
    def __init__(self) -> None:
        import os, sys
        sys.path.append('baselines/QuestEval')
        from questeval.questeval_metric import QuestEval
        self.questeval = QuestEval(no_cuda=False)

    def scorer(self, premise, hypo):
        score = self.questeval.corpus_questeval(
                hypothesis=hypo, 
                sources=premise
            )
        final_score = score['ex_level_scores']

        return torch.tensor(final_score), torch.tensor(final_score), torch.tensor(final_score)

class QAFactEvalScorer():
    def __init__(self, model_folder, device='cuda:0') -> None:
        import os, sys
        sys.path.append('baselines/QAFactEval')
        sys.path.append(os.path.abspath('baselines/qaeval/'))
        from qafacteval import QAFactEval
        kwargs = {"cuda_device": int(device.split(':')[-1]), "use_lerc_quip": True, \
                "verbose": True, "generation_batch_size": 32, \
                "answering_batch_size": 32, "lerc_batch_size": 8}

        self.metric = QAFactEval(
            lerc_quip_path=f"{model_folder}/quip-512-mocha",
            generation_model_path=f"{model_folder}/generation/model.tar.gz",
            answering_model_dir=f"{model_folder}/answering",
            lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
            lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
            **kwargs
                            )
    def scorer(self, premise, hypo):
        results = self.metric.score_batch_qafacteval(premise, [[each] for each in hypo], return_qa_pairs=True)
        score = [result[0]['qa-eval']['lerc_quip'] for result in results]
        return torch.tensor(score), torch.tensor(score), torch.tensor(score)

class MoverScorer():
    def __init__(self) -> None:
        pass

class BERTScoreFFCIScorer():
    def __init__(self) -> None:
        pass

class DAEScorer():
    def __init__(self, model_dir, device=0) -> None:
        import os, sys
        sys.path.insert(0, "baselines/factuality-datasets/")
        from evaluate_generated_outputs import daefact
        self.dae = daefact(model_dir, model_type='electra_dae', gpu_device=device)
    
    def scorer(self, premise, hypo):
        return_score = torch.tensor(self.dae.score_multi_doc(premise, hypo))

        return return_score, return_score, return_score

class SummaCScorer():
    def __init__(self, summac_type='conv', device='cuda:0') -> None:
        self.summac_type = summac_type
        import os, sys
        sys.path.append("baselines/summac")
        from summac.model_summac import SummaCZS, SummaCConv

        if summac_type == 'conv':
            self.model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=device, start_file="default", agg="mean")
        elif summac_type == 'zs':
            self.model = SummaCZS(granularity="sentence", model_name="vitc", device=device) # If you have a GPU: switch to: device="cuda"
    
    def scorer(self, premise, hypo):
        assert len(premise) == len(hypo)
        scores = self.model.score(premise, hypo)['scores']
        return_score = torch.tensor(scores)

        return return_score, return_score, return_score

class FactCCScorer():
    def __init__(self, script_path, test_data_path,result_path) -> None:
        self.script_path = script_path
        self.result_path = result_path
        self.test_data_path = test_data_path
    def scorer(self, premise, hypo):
        import subprocess
        import pickle

        self.generate_json_file(premise, hypo)
        subprocess.call(f"sh {self.script_path}", shell=True)
        print("Finishing FactCC")
        results = pickle.load(open(self.result_path, 'rb'))
        results = [-each+1 for each in results]

        return torch.tensor(results), torch.tensor(results), torch.tensor(results)
        
    def generate_json_file(self, premise, hypo):
        output = []
        assert len(premise) == len(hypo)
        i = 0
        for one_premise, one_hypo in zip(premise, hypo):
            example = dict()
            example['id'] = i
            example['text'] = one_premise
            example['claim'] = one_hypo
            example['label'] = 'CORRECT'

            i += 1
            output.append(example)
        with open(self.test_data_path, 'w', encoding='utf8') as f:
            for each in output:
                json.dump(each, f, ensure_ascii=False)
                f.write('\n')

class BLANCScorer():
    def __init__(self, device='cuda', batch_size=64) -> None:
        from blanc import BlancHelp, BlancTune
        self.blanc_help = BlancHelp(device=device, inference_batch_size=batch_size)
        

    def scorer(self, premise, hypo):
        score = self.blanc_help.eval_pairs(premise, hypo)

        return_score = torch.tensor(score)

        return return_score, return_score, return_score
        

class BLEUScorer():
    def __init__(self, n_grams=1) -> None:
        self.n_grams = n_grams
        self.n_gram_map = {
            1: (1,0,0,0),
            2: (0.5,0.5,0,0),
            3: (1./3,1./3,1./3,0),
            4: (0.25,0.25,0.25,0.25)
        }

    def scorer(self, premise, hypo):
        from nltk.translate.bleu_score import sentence_bleu
        assert len(premise) == len(hypo), "premise and hypothesis should be the same length!"

        output_score = []

        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc=f"Evaluating BLEU-{self.n_grams}", total=len(premise)):
            scores = []
            pre_sents = sent_tokenize(one_pre)
            references = [[each for each in sent.split()] for sent in pre_sents]
            for hypo_sent in sent_tokenize(one_hypo):
                hypothesis = [each for each in hypo_sent.split()]
                scores.append(sentence_bleu(references=references, hypothesis=hypothesis, weights=self.n_gram_map[self.n_grams]))
            output_score.append(sum(scores)/len(scores) if len(scores)>0 else 0.)

        return torch.tensor(output_score), torch.tensor(output_score), torch.tensor(output_score)

class ROUGEScorer():
    def __init__(self, rouge_type='1') -> None:
        from rouge import Rouge 
        self.rouge = Rouge()
        self.rouge_type = rouge_type

    def scorer(self, premise, hypo):
        
        assert len(premise) == len(hypo), "premise and hypothesis should be the same length!"

        output_score = []

        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc=f"Evaluating ROUGE-{self.rouge_type}", total=len(premise)):
            scores = []
            for pre_sent in sent_tokenize(one_pre):
                for hypo_sent in sent_tokenize(one_hypo):
                    try:
                        scores.append(self.rouge.get_scores(pre_sent, hypo_sent)[0][f"rouge-{self.rouge_type}"]['f'])
                    except:
                        if len(pre_sent.strip()) == 0:
                            print('premise sent is empty')
                        elif len(hypo_sent.strip()) == 0:
                            print('hypo sent is empty')
                        scores.append(0.0)
            scores = np.array(scores)
            scores = scores.reshape((len(sent_tokenize(one_pre)), len(sent_tokenize(one_hypo))))
            scores = scores.max(axis=0).mean()
            output_score.append(scores.item())

        return torch.tensor(output_score), torch.tensor(output_score), torch.tensor(output_score)
