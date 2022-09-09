00_rule_extraction_controller.py
	generates the rules that reconstruct BERT's predictions

00_eval.py
	evaluates the rules generated in 00_rule_extraction_controller.py

three different tasks:
AE (Aspect Extraction) -> term_type = 'aspect' in 00_... scripts
OE (Opinion/sentiment Extraction) -> term_type = 'opinion' in 00_... scripts

/rule contains py scripts used in 00_... scripts
/utils contains py scripts used in 00_... scripts

/data contains the input data (/data/input/) as well as the output rule files (/data/run/) and evaluation results (/data/run/)