#!/bin/bash

python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_ua/effortless-smoke-941 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_ua/vivid-capybara-940 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_ua/hopeful-dew-939 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_ua/cheerful-wish-1064 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Bert --snapshot results_top_ua/vibrant-chrysanthemum-1063 --batchsize 1

python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_ua/whole-breeze-974 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_ua/fragrant-salad-973 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_ua/twilight-shape-972 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_ua/red-lantern-1065 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance PubmedBert --snapshot results_top_ua/festive-cake-1066 --batchsize 1

python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_ua/twilight-lion-977 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_ua/lively-frost-976 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_ua/happy-oath-975 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_ua/floating-horse-1067 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiolinkBert --snapshot results_top_ua/enchanting-kumquat-1068 --batchsize 1

python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_ua/likely-terrain-1007 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_ua/distinctive-monkey-1006 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_ua/sleek-music-1005 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_ua/glistening-chrysanthemum-1069 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance Roberta --snapshot results_top_ua/festive-fuse-1070 --batchsize 1

python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_ua/worthy-flower-1013 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_ua/silvery-serenity-1012 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_ua/copper-planet-1011 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_ua/sparkling-envelope-1076 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRoberta --snapshot results_top_ua/abundant-envelope-1077 --batchsize 1

python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_ua/legendary-rabbit-1060 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_ua/festive-noodles-1045 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_ua/brilliant-monkey-1044 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_ua/beaming-ox-1073 --batchsize 1
python3 civic_evidence_model_evaluation.py --instance BiomedRobertaLong --snapshot results_top_ua/beaming-festival-1074 --batchsize 1

