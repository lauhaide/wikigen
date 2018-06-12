# Bootstrapping Generators from Noisy Data

Data used and implementation of the models described in [Bootstrapping Generators from Noisy Data](https://arxiv.org/abs/1804.06385). 


For any inquiry contact me at *lperez (at) ed.ac.uk*

## Dataset

Our dataset is compiled from the WikiBio dataset [(Lebret et al., 2016)](https://arxiv.org/abs/1603.07771). We use the entire abstracts and filter cases with too short/long abstracts and input property sets. The input property set is created from DBPedia. 

The base WikiBio dataset can be found [here](https://github.com/DavidGrangier/wikipedia-biography-dataset).   
The input property set extension of WikiBio can be downloaded from [here](https://drive.google.com/open?id=1jUbuyXe3R8tVQKKy5qCUh08nyBQIP3Dv).   
The pre-processed dataset to train and evaluate the content alignment model can be downloaded from [here](https://drive.google.com/open?id=1K4IyxQDD7Ui8It8qvf5MV1pqZCwUZB1b).  
The pre-processed dataset to train and evaluate the generation models can be downloaded from [here](https://drive.google.com/open?id=1CRyRWgMvymMbyqc8IHQSl6oWrH55-_uZ).  

## Models

### Content Alignment Model

The pre-trained model can be downloaded from [here](https://drive.google.com/open?id=1jZcBloHi_CShyFSapYHauqVz8HT84vPT).  
The output by the pre-trained model can be found [here](https://drive.google.com/open?id=1IUDs17e50AEhKKqkVcXVUJNg7QqyQkRh).

To train the Content Alignment model:
```
cd code/data2text-align
th train.lua -use_cuda \
-data_file ../../trained/input/dtaSDlx_LP-train.hdf5 \
-valid_data_file ../../trained/input/dtaSDlx_LP-valid.hdf5 \
-data_dict ../../trained/input/dtaSDlx_LP.dbp.dict \
-text_dict ../../trained/input/dtaSDlx_LP.text.dict \
-preembed_datavoc ../../trained/input/dtaSDlx_LP.dbp-glove.6B.200dembeddings.t7 \
-preembed_textvoc ../../trained/input/dtaSDlx_LP.text-glove.6B.200dembeddings.t7 \
-preembed -rnn_size 200 -word_vec_size 200 \
-optim adam -learning_rate 0.001 -max_batch_l 200 -min_batch_l 60 -epochs 20 \
-dataEncoder 8 -textEncoder 1
```

We tried different variants of property set and sentence encoder, those set in -dataEncoder and -textEncoder in the example are the ones described in the paper and used in the models.

#### Evaluation
```
train.lua -doeval -use_cuda \
-data_file ../../trained/input/dtaSDlx_LP-train.hdf5 \
-valid_data_file ../../trained/input/dtaSDlx_LP-valid.hdf5 \
-data_dict ../../trained/input/dtaSDlx_LP.dbp.dict \
-text_dict ../../trained/input/dtaSDlx_LP.text.dict \
-weights_file ../../trained/m_8_1scorer_weights_Wed_Sep_27_16_50_03_2017.t7 \
-max_batch_l 600 -rnn_size 200 -word_vec_size 200 -dataEncoder 8 -textEncoder 1 \
-waEvaluationSet evaluation/yawat/exist_in_select_valid-01.txt \
-distractorSet  ../../trained/input/dtaSDlx_LP-valid.distractors.dat
```
**-waEvaluationSet** provides the file with the gold manual alignments for f-score evaluation  
**-distractorSet** provides the set of 15 random distractors for each evaluated item. Note that if **-genNegValid** is passed as argument (i.e. is true), a new set of distractors will be generated and saved under the name given by argument **-distractorSet**.

The manual annotations for the word alignment measures where obtained with the Yawat tool [(Ulrich, 2008)](https://pdfs.semanticscholar.org/e747/f6af80421a278c9c6aeccb8abdf26445cb7f.pdf). If you want to look at the interface for annotations you can go here [Yawat Content Alignment](http://homepages.inf.ed.ac.uk/cgi/lperez/yawat-dta/cgi/yawat.cgi) (user=annotator1 pass=demo) .
The annotations themselves that we use can be found in *evaluation/yawat/*

This evaluation will output (files will be saved in the folder trained/results/TRAINEDMODEL/):   
- the ranking @15 score 
- the threshold for alignment selection (both *Possible* and *Sure* values [(Cohn et al., 2008)](https://www.mitpressjournals.org/doi/pdf/10.1162/coli.08-003-R1-07-044))
- it will also generate the alignment files to compute precision, recall and f-measure (next step).


##### Word Alignment Measures
```
python waq_measures.py --aFile max --resultsDir ../../trained/results/TRAINEDMODEL/
```

Compute inter-annotator agreement:
```
python waq_measures.py --iaAgreement
```

### Save Content Alignment Information

Save content alignment information for the training set (same command should be run for the validation set):

```
th dtaFiltering.lua -fold valid \
-data_file ../../trained/input/dtaSDlx_LP-valid.hdf5 \
-data_dict ../../trained/input/dtaSDlx_LP.dbp.dict \
-text_dict ../../trained/input/dtaSDlx_LP.text.dict \
-weights_file ../../trained/m_8_1scorer_weights_Wed_Sep_27_16_50_03_2017.t7 \
-rnn_size 200 -word_vec_size 200 -dataEncoder 8 -textEncoder 1 -max_batch_l 200 -min_batch_l 200 \
-wsthreshold 0.215924 -proportion 0.6 
```

This script will generate two files, one with the alignment labels and the other with the curid of each case, named *valid-alnLabels.possib_0.6.txt* and *valid-fcurid.possib_0.6.txt* . These will be placed within the results directory *trained/results/TRAINEDMODEL/filtered/*.  

### Generation models

The base code for the encoder-decoder generation models is from [(Wiseman et al, 2017)](https://arxiv.org/abs/1707.08052) (OpenNMT Torch7) and this github contains a copy of it. The base code for the Reinforcement Learning (RL) model is from [(Zhang and Lapata)](http://aclweb.org/anthology/D/D17/D17-1062.pdf). For this you will need to copy/clone from [this GitHub](https://github.com/XingxingZhang/dress) to be able to include some files under *dress/dress* directory.

The pre-trained generation models can be downloaded from [here](https://drive.google.com/open?id=1z7gG97S5DpY_NX7dLB9sl5R7jMHu1Lq-).  
The output by the pre-trained models can be found [here]().

#### Base Encoder-Decoder Generator

To train the Base Encoder-Decoder Generator, you can use the following script:

```
cd code/OpenNMT
./trainWikiOnmt.sh  GPUID 40 oAbsS_v4_aln_Sep27
```

Evaluation:

```
th wiki_train2.lua -test \
-data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-train.hdf5 \
-valid_data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-valid.hdf5 \
-test_data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-test.hdf5 \
-data_dict ../../oTrained/input/oAbsS_v4_aln_Sep27.dbp.dict \
-text_dict ../../oTrained/input/oAbsS_v4_aln_Sep27.text.dict \
-preembed_datavoc ../../vectors/oAbsS_v4_aln_Sep27.dbp-glove.6B.100dembeddings.t7 \
-train_from ../../oTrained/models/Wed_Nov__1_14_25_50_2017_epoch12_4.60.t7 \
-gen_file ../../oTrained/results/OUTDIRECTORY/output4Metrics.delex.txt \
-preembed -gpuid 1 -rnn_size 100 -word_vec_size 100 -enc_emb_size 100 -input_feed 0 -layers 1 \
-just_gen -beam_size 5 -decodingMaxLength 340
```

After running the evaluation a file is generated with the predictions by the model for the given data-set, these are saved to the *output4Metrics.delex.txt* file with a default format. After re-lexicalisation,  files for the evaluation tools can be generated from this data.


#### Multi-Task Learning Model

To train the Encoder-Decoder Generator with the Alignment Label Prediction auxiliary task, you can use the following script:

```
cd code/OpenNMT
./trainWikiOnmt.sh  GPUID 60 oAbsS_v4_aln_Sep27 \
"-guided_alignment 1 -guided_alignment_weight 0.1 -lm_objective_decay 1 -start_guided_decay 4"
```

Evaluation:

```
th wiki_train2.lua -test \
-data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-train.hdf5 \
-valid_data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-valid.hdf5 \
-test_data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-test.hdf5 \
-data_dict ../../oTrained/input/oAbsS_v4_aln_Sep27.dbp.dict \
-text_dict ../../oTrained/input/oAbsS_v4_aln_Sep27.text.dict \
-preembed_datavoc ../../vectors/oAbsS_v4_aln_Sep27.dbp-glove.6B.100dembeddings.t7 \
-train_from ../../oTrained/models/Tue_Oct_31_11_23_24_2017_epoch13_4.04.t7 \
-gen_file ../../oTrained/results/test/OUTDIRECTORY/output4Metrics.delex.txt \
-preembed -gpuid 1 -rnn_size 100 -word_vec_size 100 -enc_emb_size 100 -input_feed 0 -layers 1 \
-just_gen -beam_size 5 -decodingMaxLength 340 -guided_alignment 1
```


#### Reinforcement Learning Model

To train the Encoder-Decoder Generator with the Reinforcement Learning and the Target Content reward, you can use the following script:

```
cd code/OpenNMT
./trainWikiOnmt_RL.sh GPUID 50 "-reinforce -epochs 60 \
-train_from ../../oTrained/models/Wed_Nov__1_14_25_50_2017_epoch12_4.60.t7  -structuredCopy \
-start_epoch 1 -learning_rate 0.001 -optim sgd -learning_rate_decay 0 \
-blockRL -deltaSamplePos 3 -initialOffset 3 --enableSamplingFrom 3 -rfEpoch 2 \
-lmWeight 0 -waWeight 1 -blockWaReward  -warType 1 " oAbsS_v4_aln_Sep27
```

Evaluation

```
th wiki_train2.lua -test \
-data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-train.hdf5 \
-valid_data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-valid.hdf5 \
-test_data_file ../../oTrained/input/oAbsS_v4_aln_Sep27-test.hdf5 \
-data_dict ../../oTrained/input/oAbsS_v4_aln_Sep27.dbp.dict \
-text_dict ../../oTrained/input/oAbsS_v4_aln_Sep27.text.dict \
-preembed_datavoc ../../vectors/oAbsS_v4_aln_Sep27.dbp-glove.6B.100dembeddings.t7 \
-train_from ../../oTrained/models/Tue_Dec_19_16_52_13_2017_epoch12_4.64.t7  \
-gen_file ../../oTrained/results/test/OUTDIRECTORY/output4Metrics.delex.txt \
-preembed -gpuid 1 -rnn_size 100 -word_vec_size 100 -enc_emb_size 100 -input_feed 0 -layers 1 \
-just_gen -beam_size 5  -reinforce -max_bptt 50  -warType 1 -decodingMaxLength 340 
```


### Citation

```
@InProceedings{perez-lapata2018,
  author    = {Laura Perez-Beltrachini and Mirella Lapata},
  title     = {Bootstrapping Generators from Noisy Data},
  booktitle = {North American Chapter of the Association for Computational Linguistics},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  note = {(NAACL 2018)}
}
```