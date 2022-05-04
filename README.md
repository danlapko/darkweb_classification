## Environment installation
*`conda env create -f environment.yaml`  
*`conda activate cybersixgill_test`  
*`pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html -r requirements.txt`

# Report

## Data exploration 
Let's analyze each data column separately and look at its distribution and types.  
The analysis presented in the [data_exploration.ipynb](data_exploration.ipynb)  
<b>Main conclusions:</b>
1) We have categorical and numerical features as well as text.   
We have to build a model that uses both <b>Tabular features</b> (categorical, numerical) <b>and Unstructured Text</b>
2) We have to deal with <b>a hierarchical classification</b>
3) Dataset seems to be <b>imbalanced</b>, we have to choose appropriate metrics and approaches

## Work plan

1. Choose metrics and hierarchical classification approach;
2. <b>Baseline</b> model using only given tabular features;
3. Extract extra tabular features from text and build <b>an extensive baseline</b>;
4. Create a model that utilizes text as a sequence (e.g. transformer-based) and consumes tabular features.

### Hierarchical classification

1. Naive approach: we consider hierarchical as a multilabel classification and build two independent classifiers:  
first detects Hacking records, second detects Exploit records independently;
2. Flat approach. We create 3 types of labels and make multiclass classification: (Hacking, None), (Hacking, Exploit), None;
3. Hierarchical approach. We apply a high-level bin-classifier (Hacking) and predicted records process with a low-level classifier (Exploit).

I have tried the Naive approach and the Flat approach on the baseline model and found that the Naive works better, further I used only the Naive approach.

### Metrics
Since the dataset is unbalanced it is better to use precision, recall and f1 instead of accuracy.  
The final focus is at f1.   
[TODO] Switch to Average Precision (AP)  
Also, I use average f1 over 5-fold cross-validation.  
What about aggregation over labels: my main objective was to improve the Hacking classifier since the Exploit classifier is stable overtakes by >10%.  
It is possible to take macro or micro average but for only two classes it is easier to look at scores separately.

### Baseline

[baseline_model.ipynb](baseline_model.ipynb)  

Hacking (cross-Val K=5 avg): pr=0.6972 rec=0.5039 f1=0.5845  
Exploit (cross-Val K=5 avg): pr=0.7736 rec=0.7280 f1=0.7499

Conclusion:
The Naive approach shows better results than the Flat. Different models hit relatively close scores.


### Extensive baseline
[extensive_baseline.ipynb](extensive_baseline.ipynb)  

#### CatBoost with textual features
Text converted into BoW and used as separate categorical features. 

Hacking (cross-Val K=5 avg):  pr=0.8498 rec=0.6449 f1=0.7319  
Exploit (cross-Val K=5 avg):  pr=0.8725 rec=0.7946 f1=0.8313

#### Extra textual features (from zero-shot classification)
I've used an [open-source](https://huggingface.co/facebook/bart-large-mnli?candidateLabels=hacking%2C+fraud%2C+computer&multiClass=false&text=i+haven+t+seen+a+post+about+this+rat+lately+.+i+ve+used+it+for+testing+but+got+sick+of+it+and+deleted+it+after+.+it+s+been+around+for+a+month+but+it+is+still+unstable+and+lacking+some+features+.) zero-shot classifier to obtain extra features.  
['Hacking','Fraud','Adult'] labels used.  

[new_features_generation.py](new_features_generation.py)

Hacking (cross-Val K=5 avg): pr=0.8527 rec=0.6775 f1=0.7544    
Exploit (cross-Val K=5 avg): pr=0.8843 rec=0.7864 f1=0.8317




[TODO]:
* Use feature selection/reduction; 
* Check if CatBoost makes text preprocessing (get rid of stop-words, punctuation, dirty symbols, makes lemmatisation)
* Use GridSearch or other technics for hyper-params tuning. 
* Unsupervisely extract other features from text (e.g. BERT embeddings, Summarization).

### Sequential text-model
It would be nice to train the transformer-based model as the main classifier.

The problem is in joining unstructured text and tabular features in one model.

This type of model I had no time to implement.

I see the following approaches:
1. Split dataset in 3 parts, on the first train only BERT-like model, on the second train gradient boosting on tabular features and feature extracted from BERT (classification-score). 3rd part use for validation.
2. Bring a BERT-like model and add an extra FC layer on top of it, this layer should also consume tabular features. We can either train the whole model or freeze the original BERT and train only our HEAD.   
Interesting package [Multimodal Transformers | Transformers with Tabular Data](https://github.com/georgian-io/Multimodal-Toolkit)
3. Research more

## Conclusion

I gained the following scores:

Hacking (cross-Val K=5 avg): pr=0.8527 rec=0.6775 f1=0.7544    
Exploit (cross-Val K=5 avg): pr=0.8843 rec=0.7864 f1=0.8317

There is a lot of work to do even in an extensive baseline.  

BERT-like model with tabular features looks promising.