from evaluate import Evaluator, FLANEvaluator
from evaluate import ALL_TASKS
from baselines import *
from alignscore.inference import Inferencer
import time
import json
import os

IS_SAVE_ALL_TABLES = True
SAVE_AND_PRINT_TIMER = False

class Timer():
    def __init__(self) -> None:
        self.t0 = time.time()
        self.save_path = 'exp_results/nlg_eval_fact/time.json'
    
    def finish(self, display_name):
        t1 = time.time()
        time_pass = t1 - self.t0
        if SAVE_AND_PRINT_TIMER:
            print(f"Evalautor {display_name} finished in {time_pass} secs.")
            with open(self.save_path, 'a', encoding='utf8') as f:
                json.dump({display_name: time_pass}, f)
                f.write('\n')

class Benchmark(): 
    display_and_save_time_consumption = True
    def eval_ctc(model_type, tasks=ALL_TASKS):
        ctc_scorer = CTCScorer(model_type)
        evaluator = Evaluator(eval_tasks=tasks, align_func=ctc_scorer.score, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/CTC-{model_type}"

        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"CTC-{model_type}")

    def eval_simcse(model_type, device, tasks=ALL_TASKS):
        simcse_scorer = SimCSEScorer(model_type, device)
        evaluator = Evaluator(eval_tasks=tasks, align_func=simcse_scorer.score, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/{model_type.split('/')[-1]}_f"

        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"{model_type.split('/')[-1]}_f")

    def eval_bleurt(checkpoint, tasks=ALL_TASKS):
        bleurt_scorer = BleurtScorer(checkpoint)
        evaluator = Evaluator(eval_tasks=tasks, align_func=bleurt_scorer.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/BLEURT-20"

        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"BLEURT-20")

    def eval_bertscore(model_type, device, batch_size, tasks=ALL_TASKS):
        bertscore_scorer = BertScoreScorer(model_type=model_type, metric='f1', device=device, batch_size=batch_size)
        evaluator = Evaluator(eval_tasks=tasks, align_func=bertscore_scorer.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/bertscore_{model_type.replace('/', '-')}_f"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"bertscore_{model_type.replace('/', '-')}_f")

    def eval_bartscore(checkpoint, device, tasks=ALL_TASKS):
        bartscore_scorer = BartScoreScorer(checkpoint, device)
        evaluator = Evaluator(eval_tasks=tasks, align_func=bartscore_scorer.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/bartscore-{checkpoint.replace('/','-')}"

        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"bartscore-{checkpoint.replace('/','-')}")
    
    ### Below are Baselines for SummaC
    def eval_mnli(model="roberta-large-mnli", device='cuda:0', tasks=ALL_TASKS):
        mnli_scorer = MNLIScorer(model=model, device=device)
        evaluator = Evaluator(eval_tasks=tasks, align_func=mnli_scorer.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/mnli-{model}"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"mnli-{model}")

    def eval_ner(tasks=ALL_TASKS):
        ner_scorer = NERScorer()
        evaluator = Evaluator(eval_tasks=tasks, align_func=ner_scorer.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/NER"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"NER")
    
    def eval_unieval(tasks=ALL_TASKS, device='cuda:0'):
        unieval = UniEvalScorer(task='fact', device=device)
        evaluator = Evaluator(eval_tasks=tasks, align_func=unieval.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/UniEval"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"UniEval")
    
    def eval_feqa(tasks=ALL_TASKS):
        feqa = FEQAScorer()
        evaluator = Evaluator(eval_tasks=tasks, align_func=feqa.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/FEQA"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"FEQA")
    
    def eval_questeval(tasks=ALL_TASKS):
        questeval = QuestEvalScorer()
        evaluator = Evaluator(eval_tasks=tasks, align_func=questeval.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/QuestEval"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"QuestEval")
    
    def eval_qafacteval(tasks=ALL_TASKS, device='cuda:0'):
        import os, sys
        warning("using conda env qaeval!!!")
        qafacteval = QAFactEvalScorer(device=device, model_folder=os.path.abspath('../BaselineForNLGEval/QAFactEval/models'))
        evaluator = Evaluator(eval_tasks=tasks, align_func=qafacteval.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/QAFactEval"
        evaluator.evaluate()

    def eval_dae(tasks=ALL_TASKS, model_dir=None, device=0):
        dae = DAEScorer(model_dir=model_dir, device=device)
        evaluator = Evaluator(eval_tasks=tasks, align_func=dae.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/DAE"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"DAE")

    def eval_bleu(tasks=ALL_TASKS, n_grams=1):
        bleu = BLEUScorer(n_grams=n_grams)
        evaluator = Evaluator(eval_tasks=tasks, align_func=bleu.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/BLEU-{n_grams}"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"BLEU-{n_grams}")

    def eval_rouge(tasks=ALL_TASKS, rouge_type='1'):
        rouge = ROUGEScorer(rouge_type=rouge_type)
        evaluator = Evaluator(eval_tasks=tasks, align_func=rouge.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/ROUGE-{rouge_type}"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"ROUGE-{rouge_type}")

    def eval_factcc(script_path, test_data_path,result_path, tasks=ALL_TASKS):
        factcc = FactCCScorer(script_path=script_path, test_data_path=test_data_path, result_path=result_path)
        evaluator = Evaluator(eval_tasks=tasks, align_func=factcc.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/FactCC"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"FactCC")

    def eval_blanc(tasks=ALL_TASKS, device='cuda:0', batch_size=64):
        blanc = BLANCScorer(device=device, batch_size=batch_size)
        evaluator = Evaluator(eval_tasks=tasks, align_func=blanc.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/BLANC"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"BLANC")
    
    def eval_summac(tasks=ALL_TASKS, summac_type='conv', device='cuda:0'):
        summac = SummaCScorer(summac_type=summac_type, device=device)
        evaluator = Evaluator(eval_tasks=tasks, align_func=summac.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/baselines/SummaC-{summac_type}"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"SummaC-{summac_type}")

    def eval_flan(tasks=ALL_TASKS, device='cuda:0', model_name='google/flan-t5-base', batch_size=32):
        flan_t5 = FLANScorer(device=device, model_name=model_name, batch_size=batch_size)
        evaluator = FLANEvaluator(eval_tasks=tasks, align_func=flan_t5.scorer, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"align_eval/baselines/FLAN-{model_name.replace('/', '-')}"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"FLAN-{model_name.replace('/', '-')}")
    
    def eval_align_nlg(ckpt_path, ablation='full', base_model='roberta-large', batch_size=32, device='cuda:0', tasks=ALL_TASKS, nlg_eval_mode='nli_sp'):
        align = Inferencer(ckpt_path=ckpt_path, model=base_model, batch_size=batch_size, device=device)
        align.nlg_eval_mode = nlg_eval_mode

        evaluator = Evaluator(eval_tasks=tasks, align_func=align.nlg_eval, is_save_all_tables=IS_SAVE_ALL_TABLES)
        # evaluator.result_save_name = f"nlg_eval_fact/align/ALIGN-{nlg_eval_mode}-{base_model}" if ablation == 'full' else f"nlg_eval_fact/ablation/ALIGN-{nlg_eval_mode}-{base_model}-{ablation}"
        evaluator.result_save_name = f"align_eval/more-qa/ALIGNScore-{nlg_eval_mode}-{base_model}"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"ALIGN-{nlg_eval_mode}-{base_model}" if ablation == 'full' else f"ALIGN-{nlg_eval_mode}-{base_model}-{ablation}")

    def eval_align_nlg_smart(ckpt_path, smart_type='smart-n', base_model='roberta-large', batch_size=32, device='cuda:0', tasks=ALL_TASKS):
        align = Inferencer(ckpt_path=ckpt_path, model=base_model, batch_size=batch_size, device=device)
        align.smart_type = smart_type

        evaluator = Evaluator(eval_tasks=tasks, align_func=align.smart_doc, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"nlg_eval_fact/ablation/ALIGN-{smart_type}-{base_model}"
        
        timer = Timer()
        evaluator.evaluate()
        timer.finish(f"ALIGN-{smart_type}-{base_model}")

    def eval_align(ckpt_path, save_comment="", align_type='align-wo-finetune', base_model='roberta-large', batch_size=32, device='cuda:0', tasks=ALL_TASKS):
        align = Inferencer(ckpt_path=ckpt_path, model=base_model, batch_size=batch_size, device=device)

        evaluator = Evaluator(eval_tasks=tasks, align_func=align.inference, is_save_all_tables=IS_SAVE_ALL_TABLES)
        evaluator.result_save_name = f"align_eval/{align_type}/{save_comment}{ckpt_path.split('/')[-1].split('.ckpt')[0]}" 
        
        evaluator.evaluate()

    if __name__ == "__main__": 
        FACT_EVAL_TASKS = ['summac', 'true','xsumfaith', 'summeval', 'qags_xsum', 'qags_cnndm', 'newsroom', 'rank19', 'frank', 'samsum']

        eval_ctc('D-cnndm', tasks=FACT_EVAL_TASKS)
        eval_simcse('princeton-nlp/sup-simcse-roberta-large', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_bleurt('data/bleurt_checkpoints/BLEURT-20', tasks=FACT_EVAL_TASKS) # Use CUDA_VISIBLE_DEVICES=7 python benchmark.py to compile
        eval_bertscore('microsoft/deberta-xlarge-mnli', device='cuda:0', tasks=FACT_EVAL_TASKS, batch_size=16)
        eval_bartscore('facebook/bart-large-cnn', device=0, tasks=FACT_EVAL_TASKS)
        eval_mnli(model="roberta-large-mnli", device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_ner(tasks=FACT_EVAL_TASKS)
        eval_unieval(tasks=FACT_EVAL_TASKS, device='cuda:0')

        eval_feqa(tasks=FACT_EVAL_TASKS)
        eval_questeval(tasks=FACT_EVAL_TASKS)

        eval_qafacteval(tasks=FACT_EVAL_TASKS) #####

        eval_bleu(tasks=FACT_EVAL_TASKS, n_grams=1)
        eval_bleu(tasks=FACT_EVAL_TASKS, n_grams=2)
        eval_bleu(tasks=FACT_EVAL_TASKS, n_grams=3)
        eval_bleu(tasks=FACT_EVAL_TASKS, n_grams=4)
        eval_rouge(tasks=FACT_EVAL_TASKS, rouge_type=1)
        eval_rouge(tasks=FACT_EVAL_TASKS, rouge_type=2)
        eval_rouge(tasks=FACT_EVAL_TASKS, rouge_type='l')

        eval_dae(tasks=FACT_EVAL_TASKS, model_dir=os.path.abspath("../BaselineForNLGEval/factuality-datasets/DAE_xsum_human_best_ckpt"))

        eval_factcc(tasks=FACT_EVAL_TASKS, script_path=os.path.abspath("../BaselineForNLGEval/factCC/modeling/scripts/factcc-eval.sh"), test_data_path=os.path.abspath("../BaselineForNLGEval/factCC/data/data-dev.jsonl"), result_path=os.path.abspath("../BaselineForNLGEval/factCC/factcc-checkpoint/eval_results.pkl"))
        eval_blanc(tasks=FACT_EVAL_TASKS, device='cuda:0', batch_size=64)
        eval_summac(tasks=FACT_EVAL_TASKS, device='cuda:0', summac_type='conv')
        eval_summac(tasks=FACT_EVAL_TASKS, device='cuda:0', summac_type='zs')

        ## OUR Model
        # Roberta-base
        eval_align_nlg(nlg_eval_mode='bin',ablation='full', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='bin_sp',ablation='full', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli',ablation='full', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='full', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='reg',ablation='full', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='reg_sp',ablation='full', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)

        # Roberta-large
        eval_align_nlg(nlg_eval_mode='bin',ablation='full', ckpt_path="checkpoints/roberta-large/roberta-large_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x4x1_final.ckpt", base_model='roberta-large', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='bin_sp',ablation='full', ckpt_path="checkpoints/roberta-large/roberta-large_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x4x1_final.ckpt", base_model='roberta-large', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli',ablation='full', ckpt_path="checkpoints/roberta-large/roberta-large_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x4x1_final.ckpt", base_model='roberta-large', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='full', ckpt_path="checkpoints/roberta-large/roberta-large_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x4x1_final.ckpt", base_model='roberta-large', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='reg',ablation='full', ckpt_path="checkpoints/roberta-large/roberta-large_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x4x1_final.ckpt", base_model='roberta-large', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='reg_sp',ablation='full', ckpt_path="checkpoints/roberta-large/roberta-large_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x4x1_final.ckpt", base_model='roberta-large', device='cuda:0', tasks=FACT_EVAL_TASKS)

        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='full', ckpt_path="checkpoints/more-qa-scale-loss/roberta-large/roberta-large_no_mlm_full-dataset_500000_32x4x1_final.ckpt", base_model='roberta-large', device='cuda:5', tasks=FACT_EVAL_TASKS)

        # Ablation based on Roberta-base
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='fv', ckpt_path="checkpoints/ablation/no-fv/roberta-base/roberta-base_no_mlm_mnli_doc_nli_squad_v2_paws_paws_qqp_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='ir', ckpt_path="checkpoints/ablation/no-ir/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='nli', ckpt_path="checkpoints/ablation/no-nli/roberta-base/roberta-base_no_mlm_nli_fever_squad_v2_paws_paws_qqp_vitaminc_race_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='para', ckpt_path="checkpoints/ablation/no-para/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_wiki103_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='qa', ckpt_path="checkpoints/ablation/no-qa/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_paws_paws_qqp_vitaminc_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='sts', ckpt_path="checkpoints/ablation/no-sts/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg(nlg_eval_mode='nli_sp',ablation='unsup', ckpt_path="checkpoints/ablation/no-unsup/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_msmarco_paws_unlabeled_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg_smart(smart_type='smart-l', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
        eval_align_nlg_smart(smart_type='smart-n', ckpt_path="checkpoints/roberta-base/roberta-base_no_mlm_mnli_nli_fever_doc_nli_squad_v2_paws_paws_qqp_vitaminc_race_anli_r1_anli_r2_anli_r3_snli_wikihow_msmarco_paws_unlabeled_wiki103_qqp_stsb_sick_500000_32x2x1_final.ckpt", base_model='roberta-base', device='cuda:0', tasks=FACT_EVAL_TASKS)
