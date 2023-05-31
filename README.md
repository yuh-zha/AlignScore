# AlignScore
[AlignScore: Evaluating Factual Consistency with a Unified Alignment Function](https://arxiv.org/abs/2305.16739)

by Yuheng Zha, Yichi Yang, Ruichen Li and Zhiting Hu

Accepted at ACL 2023 
# Usage
## Evaluate Factual Consistency
To evaluate the factual consistency of the `claim` w.r.t. the `context`, simply use the score function from `alignscore`.
```python
from alignscore import alignscore

scorer = alignscore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path:path_to_checkpoint, evaluation_mode='nli_sp')
score = scorer.score(context=['hello world'], claim=['hello world'])
```
`model`: the backbone model of the metric. Now, we only provide the metric trained on RoBERTa

`batch_size`: the batch size of inference. Larger batch size may accelerate the 


## Checkpoints
We provide two checkpoints in the paper: `AlignScore-base` and `AlignScore-large`. The `-base` model was trained on RoBERTa-base and has 125M parameters. The `-large` model was trained on RoBERTa-large and has 355M parameters. 

The link is shown below:

**AlignScore-base**: 
https://drive.google.com/file/d/1e0U_Qo8V4s8PPM-8eUBT0-SQhbYXJCVd/view?usp=sharing

**AlignScore-large**:
https://drive.google.com/file/d/1juDipD7EDsyFSyJs8nzyHuPpedfdoRgp/view?usp=sharing


# Leaderboard
| Rank | Metrics          | SummaC | TRUE | Other-Spearman | Average |
| ---- | :--------------- | :----: | :--: | :------------: | :-----: |
| 1    | AlignScore-large |  88.6  | 83.8 |      49.3      |  73.9   |
| 2    | AlignScore-base  |  87.4  | 82.5 |      44.9      |  71.6   |
| 3    | QAFactEval       |  83.8  | 79.4 |      42.4      |  68.5   |
| 4    | UniEval          |  84.6  | 78.0 |      41.5      |  68.0   |
| 5    | SummaC-CONV      |  81.0  | 78.7 |      34.2      |  64.6   |
| 6    | BARTScore        |  80.9  | 73.4 |      34.8      |  63.0   |
| 7    | CTC              |  81.2  | 72.4 |      35.3      |  63.0   |
| 8    | SummaC-ZS        |  79.0  | 78.2 |      30.4      |  62.5   |
| 9    | ROUGE-2          |  78.1  | 72.4 |      27.9      |  59.5   |
| 10   | ROUGE-1          |  77.4  | 72.0 |      28.6      |  59.3   |
| 11   | ROUGE-L          |  77.3  | 71.8 |      28.3      |  59.1   |
| 12   | QuestEval        |  72.5  | 71.4 |      25.0      |  56.3   |
| 13   | BLEU             |  76.3  | 67.3 |      24.6      |  56.1   |
| 14   | DAE              |  66.8  | 65.7 |      35.1      |  55.8   |
| 15   | BLEURT           |  69.2  | 71.9 |      24.9      |  55.4   |
| 16   | BERTScore        |  72.1  | 68.6 |      21.9      |  54.2   |
| 17   | SimCSE           |  67.4  | 70.3 |      23.8      |  53.8   |
| 18   | FactCC           |  68.8  | 62.7 |      21.2      |  50.9   |
| 19   | BLANC            |  65.1  | 64.0 |      14.4      |  47.8   |
| 20   | NER-Overlap      |  60.4  | 59.3 |      18.9      |  46.2   |
| 21   | MNLI             |  47.9  | 60.4 |      3.1       |  37.2   |
| 22   | FEQA             |  48.3  | 52.2 |      -1.9      |  32.9   |
