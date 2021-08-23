# Thesis Bias in NLP 

This repository holds the source code and data used in Angelie Kraft's Master's thesis (inovex 
GmbH & University of Hamburg). The thesis' title is "Triggering Models: Measuring and 
Mitigating Bias in German Language Generation". It replicates the work by Sheng et al. (2019) 
and Sheng et al. (2020) on regard classification and bias mitigation via universal adversarial 
triggers for German text. 

The examples and scripts use GPT-3 and GerPT-2 (https://huggingface.co/benjamin/gerpt2-large) for 
generative tasks. Some preliminary explorations were done with GPT-Neo.


# The data

Different development data and experiment artifacts are included in the `data` folder:

For training and evaluation of the classifier:
* `annotated_data_raw/crowd_sourced_regard_w_annotations`: crowd-sourced human-authored dataset 
  with human annotations
* `annotated_data_raw/gerpt2_generated_regard_w_annotations`: GerPT-2-generated dataset with 
  human annotations
* `raw_study_data`: the raw crowd-sourced data as downloaded from the survey.

For trigger search:
* `trigger_search_data_preprocessed`: this is the GerPT-2-generated, human-labeled data from 
  before but processed to be applicable for trigger search 

Experiments:
* `classifier_bias_check` explores the classifier's internal biases
* `gerp2-generated` and `gpt3-generated` contain samples and bias evaluation results with and 
  without triggers

# The source code
The scripts, notebooks, and data provided here intend to allow an exploration of bias and 
debiasing effects through the triggers. Due to this exploratory nature, there are various 
modes and options provided here. Most things can be done via `python run.py run_mode=MODENAME`. 
However, it is recommended to check out the detailed options within the respective config files.

## Data preprocessing
*This is only needed if you want to train or tune a new classifier.*
The preprocessed and pre-embedded data from the thesis are also provided with this repository.

To preprocess data from the annotated datasets in 
`data/annotated_data_raw/crowd_sourced_regard_w_annotations` with `run_mode=data`.

In `conf/config.yaml` check `dev_settings` for additional settings. Further, the settings for 
`classifier`, `embedding`, and `preprocessing` should be adjusted, depending on the type of 
classifier you want to train.

Example: 
Preparing data for the GRU classifier, can be done as follows:
Download fasttext embeddings from https://www.deepset.
ai/german-word-embeddings. Then run `python run.py run_mode=data 
classifier=lstm embedding=fastt preprocessing=for_lstm` (the gru unit type is specified in the 
classifier settings). 

### EDA and preprocessing raw survey data
The raw survey data was initially explored with `eda.ipynb`. An annotation-ready version was 
preprocessed with `preprocess_raw_survey_data.ipynb`. 

## Regard classifier
The pretrained SentenceBERT-based regard classifier is stored in 
`model/sbert_regard_classifier.pth`.
To do things with the classifier, set `run_mode=classifier` in `conf/config.yaml`.
In folder `conf/classifier_mode` you can see the different supported modes. 

You can do the following things:
* train it with incremental dataset sizes (to analyze data 
requirements)
* tune a new classifier to find the best hyperparameters
* train a new classifier on predefined hyperparameters 
* evaluate a pretrained classifier on a test set
* predict the regard for a list of texts with a pretrained classifier (if the given data comes 
  with a "Label" column, evaluation scores are computed)

In `conf/config.yaml` you can choose which type of classifier and embedding you want to use. 
Make sure to look at the respective yaml-files to learn more about your setting options. E.g., 
the `train.yaml`

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

Again stdout is redirected to a log file in `outputs`.

### Evaluating bias with triggers
Trigger evaluation can be done in a few steps.

1. Generate data with trigger by using `run_mode=generate` and specifying the trigger in 
   `conf/run_mode/generate.yaml`. You may change the generative LM in `conf/config.yaml`. The 
   respective generator's settings allow changing the sampling parameters.
2. Classify the generated sentences with the pretrained regard classifier via 
   `run_mode=classifier` and `classifier_mode=predict`.
3. Finally, run `run_mode=eval_bias` after specifying the paths in `conf/run_mode/eval_bias.yaml`.

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