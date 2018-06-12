CUDA_VISIBLE_DEVICES=$1 th wiki_train2.lua -gpuid 1 \
-data_file ../../oTrained/input/$3-train.hdf5 \
-valid_data_file ../../oTrained/input/$3-valid.hdf5 \
-data_dict ../../oTrained/input/$3.dbp.dict \
-text_dict ../../oTrained/input/$3.text.dict \
-preembed -preembed_datavoc ./../oTrained/input/$3.dbp-glove.6B.100dembeddings.t7  \
-rnn_size 100 -word_vec_size 100 -enc_emb_size 100 -input_feed 0 -layers 1 \
-epochs 20 -learning_rate 0.001 -optim adam \
-save_model ../../oTrained/models/ -max_bptt $2 $4 "
