# Bootstrapping Generators from Noisy Data

Data used and implementation of the models described in [Bootstrapping Generators from Noisy Data](https://arxiv.org/abs/1804.06385). 


For any inquiry contact me at *lperez (at) ed.ac.uk*

## Dataset

Our dataset is compiled from the WikiBio dataset [(Lebret et al., 2016)](https://arxiv.org/abs/1603.07771). We use the entire abstracts and filter cases with too short/long abstracts and input property sets. The input property set is created from DBPedia. 

The extended data-set can be downloaded from [here](). 
The pre-processed dataset files to train and evaluate the models can be downloaded from [here]().

## Models
The base code for the encoder-decoder generation models is from [(Wiseman et al, 2017)](https://arxiv.org/abs/1707.08052), based in turn on Torch7 OpenNMT. The base code for the Reinforcement Learning (RL) model is from [(Zhang and Lapata)](http://aclweb.org/anthology/D/D17/D17-1062.pdf).


### Content Alignment Model

To train the Content Alignment model:
```
th train.lua -use_cuda \
-data_file PATHPREPROCFILES/dta-train.hdf5 \
-valid_data_file PATHPREPROCFILES/dta-valid.hdf5 \
-data_dict PATHPREPROCFILES/dta.dbp.dict \
-text_dict PATHPREPROCFILES/dta.text.dict \
-preembed_datavoc PATHPREPROCFILES/dta.dbp-glove.6B.200dembeddings.t7 \
-preembed_textvoc PATHPREPROCFILES/dta.text-glove.6B.200dembeddings.t7 \
-preembed -rnn_size 200 -word_vec_size 200 \
-optim adam -learning_rate 0.001 -max_batch_l 200 -min_batch_l 60 -epochs 20 \
-dataEncoder 8 -textEncoder 1
```

We tried different variants of property set and sentence encoder, those set in -dataEncoder and -textEncoder in the example are the ones described in the paper and used in the models.

#### Evaluation
```
train.lua -doeval -use_cuda \
-data_file PATHPREPROCFILES/dta-train.hdf5 \
-valid_data_file PATHPREPROCFILES/dta-valid.hdf5 \
-data_dict PATHPREPROCFILES/dta.dbp.dict \
-text_dict PATHPREPROCFILES/dta.text.dict \
-weights_file TRAINEDMODEL \
-max_batch_l 600 -rnn_size 200 -word_vec_size 200 -dataEncoder 8 -textEncoder 1 \
-waEvaluationSet evaluation/yawat/exist_in_select_valid-01.txt \
-distractorSet  evaluation/valid-distrSet-dtaAbsS_aln_v4.dat
```
-waEvaluationSet provides the file with the gold manual alignments for f-score evaluation  
-distractorSet provides the set of 15 random distractors for each evaluated item. Note that if the argument -genNegValid is on the command line a new set of distractors will be generated and saved under that name.

This evaluation will output (files will be saved in the folder trained/results/TRAINEDMODEL/):   
- the ranking @15 score 
- the threshold for alignment selection (both *Possible* and *Sure* values [(Cohn et al., 2008)](https://www.mitpressjournals.org/doi/pdf/10.1162/coli.08-003-R1-07-044))
- it will also generate the alignment files to compute precision, recall and f-measure (next step Word Alignment Measures)


##### Word Alignment Measures
```
python waq_measures.py --aFile max --resultsDir trained/results/TRAINEDMODEL/
```

Compute inter-annotator agreement:
```
python waq_measures.py --iaAgreement
```

### Save Content Alignment Information

Save content alignment information for the training set (same command should be run for the validation set):

```
th dtaFiltering.lua -fold valid \
-data_file ../../trained/input/dta-valid.hdf5 \
-data_dict ../../trained/input/dta.dbp.dict \
-text_dict ../../trained/input/dta.text.dict \
-weights_file ../../trained/MOEDEL  \
-rnn_size 200 -word_vec_size 200 -dataEncoder 8 -textEncoder 1 -max_batch_l 200 -min_batch_l 200 \
-wsthreshold 0.488366 -proportion 0.6 
```


### Next sections and other stuff will be upload soon, please check again the repo for new things!

#### Base Encoder-Decoder Generator

#### Multi-Task Learning Model

#### Reinforcement Learning Model

