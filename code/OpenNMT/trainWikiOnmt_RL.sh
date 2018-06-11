
CUDA_VISIBLE_DEVICES=$1 th wiki_train2.lua -gpuid 1\
-data_file ../../oTrained/input/$4-train.hdf5 \
-valid_data_file ../../oTrained/input/$4-valid.hdf5 \
-data_dict ../../oTrained/input/$4.dbp.dict \
-text_dict ../../oTrained/input/$4.text.dict \
-preembed -preembed_datavoc ../../vectors/$4.dbp-glove.6B.100dembeddings.t7  \
-rnn_size 100 -word_vec_size 100 -enc_emb_size 100 -input_feed 0 -layers 1 -max_bptt $2 $3 \
-save_model ../../oTrained/models/
