#!/bin/bash

python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_mc/ablaze-heart-1203 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_mc/angelic-chocolate-1204 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_mc/entrancing-heartthrob-1109 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_mc/candlelit-heartthrob-1108 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_mc/expressive-cupid-1107 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL

python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_mc/euphoric-flower-1206 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_mc/delirious-date-1205 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_mc/beguiling-caress-1121 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_mc/spellbinding-hug-1120 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_mc/honest-etchings-1119 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL

python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_mc/warm-balloon-1208 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_mc/dreamy-dove-1207 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_mc/hunky-heart-1145 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_mc/ecstatic-crush-1144 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_mc/starry-eyed-candles-1143 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL

python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_mc/doting-violet-1210 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_mc/enchanting-chocolate-1209 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_mc/sweet-tulip-1202 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_mc/daring-tulip-1201 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_mc/gentle-hug-1200 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL

python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_mc/kind-sweetheart-1212 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_mc/blazing-candles-1211 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_mc/ablaze-chocolate-1163 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_mc/sparkling-dove-1162 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_mc/constant-candy-heart-1161 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL

python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_mc/daring-caress-1215 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_mc/appealing-infatuation-1213 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_mc/forthright-smooch-1181 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_mc/radiant-lovebird-1180 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_mc/expressive-date-1179 --batchsize 1 --mode ABSTRACTS_ONLY_MULTILABEL
