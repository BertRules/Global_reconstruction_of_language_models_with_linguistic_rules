# Description
00_rule_extraction_controller.py
	generates the rules that reconstruct BERT's predictions

00_eval.py
	evaluates the rules generated in 00_rule_extraction_controller.py

The following two token classification tasks are focused on for global reconstruction of language models:
- AE (Aspect Extraction) -> term_type = 'aspect' in 00_...py scripts
- OE (Opinion/sentiment Extraction) -> term_type = 'opinion' in 00_...py scripts

Description of the folders:
- '/rule' contains py scripts used in 00_...py scripts
- '/utils' contains py scripts used in 00_...py scripts
- '/data' contains the input data ('/data/input/') as well as the output rule files ('/data/run/') and evaluation results ('/data/run/')
- To access the input data, please read the file '/data/input/README.md'

# Note
This project is based on the code from the paper Dai, H., & Song, Y. (2019). Neural aspect and opinion term extraction with mined rules as weak supervision (cf. https://aclanthology.org/P19-1520/), which can be found here: https://github.com/HKUST-KnowComp/RINANTE
