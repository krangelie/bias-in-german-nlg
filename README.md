# Thesis Bias in NLP 

This repository holds the source code and data used in Angelie Kraft's Master's thesis (inovex 
GmbH & University of Hamburg). The thesis' title is "Triggering Models: Measuring and 
Mitigating Bias in German Language Generation". It replicates the work by Sheng et al. (2019) 
and Sheng et al. (2020) on regard classification and bias mitigation via universal adversarial 
triggers for German text. 

The examples and scripts use GPT-3 and GerPT-2 (https://huggingface.co/benjamin/gerpt2-large) for 
generative tasks. Some preliminary explorations were done with GPT-Neo.

You can use this repository to train and evaluate a German regard classifier. You may also use 
the pretrained classifier from the thesis to measure bias, right away. Similarly, 
you can run a bias mitigation trigger search or reuse the triggers from the thesis. Detailed 
descriptions below. (Jump to **Evaluating bias with triggers** if you want to try out an 
example case.)

# The data

Different development data and experiment artifacts are included in the `data` folder:

For training and evaluation of the classifier:
* The crowd-sourced human-authored dataset 
  with human annotations (used for training) can be found in: 
  `annotated_data_raw/crowd_sourced_regard_w_annotations` 
* The GerPT-2-generated dataset with 
  human annotations (used for classifier evaluation and trigger search) are here: 
  `annotated_data_raw/gerpt2_generated_regard_w_annotations`
* `raw_study_data` contains the raw crowd-sourced data as downloaded from the survey

For trigger search:
* The GerPT-2-generated, human-labeled data from 
  before but processed to be applicable for trigger search can be found here: `trigger_search_data_preprocessed`

Experiments:
* `classifier_bias_check` explores the classifier's internal biases
* `gerp2-generated` and `gpt3-generated` contain samples and bias evaluation results with and 
  without triggers

**Warning: Some samples are explicit or offensive in nature.**

# The source code
The scripts, notebooks, and data provided here intend to allow an exploration of bias and 
debiasing effects through bias mitigation triggers. Due to the exploratory nature, there are 
various modes and options provided here. Switching between modes can be done via `python run.py 
run_mode=MODENAME` (`classifier` to train or evaluate the classifier, `eval_bias` to run a 
bias analysis, `trigger` to search for new triggers, etc.). 
It is definitely recommended checking out the detailed options within the respective config files.

## Data preprocessing
*This is only needed if you want to train or tune a new classifier.*
The preprocessed and pre-embedded data from the thesis are also provided with this repository.

Preprocess data from the annotated datasets in 
`data/annotated_data_raw/crowd_sourced_regard_w_annotations` with `run_mode=data`.
Before running the script, make sure to check out `conf/config.yaml` for `dev_settings`, 
`classifier`, `embedding`, and `preprocessing`. They should be adjusted, depending on the type of 
classifier you want to train.

Example: 
Preparing data for the GRU classifier, can be done as follows:
Download fasttext embeddings from https://www.deepset.ai/german-word-embeddings. Store the 
`model.bin` in `models/fasttext/`. Then run `python run.py run_mode=data 
classifier=lstm embedding=fastt preprocessing=for_lstm dev_settings.annotation=majority` (the gru 
unit type is specified in the 
classifier settings). 

### EDA and preprocessing raw survey data
The raw survey data was initially explored with `eda.ipynb`. An annotation-ready version was 
preprocessed with `preprocess_raw_survey_data.ipynb`. 

## Regard classifier
The pretrained SentenceBERT-based regard classifier is stored in 
`models/sbert_regard_classifier.pth`. 


You can do the following with `run_mode=classifier`:

* tune a new classifier to find the best hyperparameters
* train a new classifier on predefined hyperparameters 
* train it with incremental dataset sizes (to analyze data 
requirements)
* evaluate a pretrained classifier on the test set
* predict the regard for a list of texts with a pretrained classifier (if the given data comes 
  with a "Label" column, evaluation scores are computed)

Example:
To train a GRU classifier run `python run.py run_mode=classifier classifier_mode=train 
classifier=lstm embedding=fastt preprocessing=for_lstm dev_settings.annotation=majority`.

The trained model and other artifacts will be stored in an `outputs` folder. Note that most 
scripts redirect stdout to a log file within this folder.

## Bias mitigation with universal adversarial triggers

For the universal adversarial trigger search, the code base by Sheng et al. (2020; 
https://github.com/ewsheng/controllable-nlg-biases) was used. The scripts were adjusted 
and refactored for this project. 

###  Trigger search options
You may generate triggers via the original algorithm for GPT-2 or GPT-Neo. 
See config file `conf/run_mode/trigger.yaml` for your search options. 

Alternatively, a naive trigger search was implemented, too. Respective options can be 
found in `conf/run_mode/naive_trigger.yaml`

Again stdout is redirected to a log file in `outputs`. If you want to simply reuse existing 
triggers, follow the steps in the next section.

### Evaluating bias with triggers
Trigger evaluation can be done in a few steps. 

1. Generate data with trigger with `python run.py run_mode=generate gpt=gpt2`. You 
   can specify the trigger in `conf/run_mode/generate.yaml`. If you want to change the 
   generator settings see `conf/gpt/gpt2.yaml`.
4. Classify the generated sentences with the pretrained regard classifier via 
   `python run.py run_mode=classifier classifier_mode=predict`. 
5. Finally, run `python run.py run_mode=eval_bias`.

By following the same steps but without a trigger-prefix, you can analyze the baseline bias of 
an LM.

#### Additional analyses and notebooks

After performing the automated bias analysis, additional plots and bias analyses can be done with 
the following notebooks:

`occupation_stereotypes.ipynb`, `plot_rel_regard_changes.ipynb`, `ratio_plot.ipynb`


# References
* Sheng, E., Chang, K. W., Natarajan, P., & Peng, N. (2019). The Woman Worked as a Babysitter: On 
 Biases in Language Generation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 3407-3412).
* Sheng, E., Chang, K. W., Natarajan, P., & Peng, N. (2020). Towards Controllable Biases in 
  Language Generation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings (pp. 3239-3254).