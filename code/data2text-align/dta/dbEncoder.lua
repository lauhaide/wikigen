--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 06/06/17
-- Time: 15:07
-- To change this template use File | Settings | File Templates.
--



local dbEncoder = {}
--local embeddingl = require('dta.embeddingl') --TODO: this is different when used from this project than  when used from onmt == see to make same
local embeddingl = require('dta.embeddingl')

function dbEncoder.build(opt, dataset, voc_data_size, train)

  local dataEnc = nil
  if opt.dataEncoder==1 then
    print("structured triple encoding **triple elements concat**...")
    dataEnc = dbEncoder.getDataEncoderStruct(opt, voc_data_size, dataset.max_prop_namel, dataset.max_prop_l, dataset.max_dataseq_l, train)
  elseif opt.dataEncoder==2 then
    print("triple BOW *hirarchical* encoding...")
    dataEnc = dbEncoder.getDataEncoderOneSequenceBOW_2D(opt, voc_data_size, train)
  elseif  opt.dataEncoder==3 then
    print("property name only as BOW *single flattened multi-sequence* encoding...")
    dataEnc = dbEncoder.getDataEncoderOneSequenceBOW_1D(opt, voc_data_size, dataset.max_prop_namel, dataset.max_dataseq_l, train)
  elseif  opt.dataEncoder==4 then
    print("triple BOW *single flattened multi-sequence* encoding...")
    dataEnc = dbEncoder.getDataEncoderOneSequenceBOW_1D(opt, voc_data_size, dataset.max_prop_l, dataset.max_dataseq_l, train)
  elseif opt.dataEncoder==5 then
    print("triple sequence LSTM *hirarchical* encoding...")
    dataEnc = dbEncoder.getDataEncoderOneSequenceLSTM_2D(opt, voc_data_size, train)
  elseif opt.dataEncoder==6 then
    print("triple sequence LSTM *single flattened multi-sequence* encoding...")
    dataEnc = dbEncoder.getDataEncoderOneSequenceLSTM_1D(opt, voc_data_size, dataset.max_prop_l, dataset.max_dataseq_l, train)
  elseif opt.dataEncoder==7 then
    print("structured triple encoding ** triple elements concat + linear **...")
    dataEnc = dbEncoder.getDataEncoderStruct_7(opt, voc_data_size, dataset.max_prop_namel, dataset.max_prop_l, dataset.max_dataseq_l, train)
  elseif opt.dataEncoder==8 then
    print("triple sequence biLSTM *hirarchical* encoding...")
    dataEnc = dbEncoder.getDataEncoderOneSequenceBiLSTM_2D(opt, voc_data_size, train)
  end
  return dataEnc
end


function dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local emb_data
  if not train then
    emb_data = nn.LookupTableMaskZero(voc_data_size, opt.word_vec_size)
  else
    if opt.preembed then
      --load pre-trained word vectors
      emb_data = embeddingl.initPreTrainedEmbeddingMatrix(voc_data_size, opt.word_vec_size, opt.preembed_datavoc)
    else
      emb_data = embeddingl.initRandomEmbeddingMatrix(voc_data_size, opt.word_vec_size)
    end
  end
  return emb_data
end



function dbEncoder.getDataEncoderOneSequenceBiLSTM_2D(opt, voc_data_size, train)

  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local enc = nn.Sequential()
  local seqlstm = nn.Sequential()
  local lstm = nn.SeqBRNN(opt.word_vec_size, opt.rnn_size, true) --TODO: revise all RNNs throughout input and output sizes can be different!!!
  lstm.maskzero = true
  seqlstm:add(nn.Contiguous())
  seqlstm:add(lstm)
  seqlstm:add(nn.Select(2,-1))
  local bemb = nn.Bottle(emb_data, 2,3)
  local brnn = nn.Bottle(seqlstm, 3,2)

  enc:add(bemb)
  enc:add(brnn)

  return localize(enc)
end


--encodes property-name property-value sequences as composed f(property-name, property-value) with linear composition
function dbEncoder.getDataEncoderStruct_7(opt, voc_data_size, propNameLength, propLength, dataPropSeqLength, train)
  --[[encode input property set]]

  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local emb_data_val = emb_data:clone('weight', 'bias', 'gradWeight', 'gradBias')

  local data_enc = nn.Sequential()
  local component_data = nn.ParallelTable()

  local property
  if opt.propNameKB then
    --TODO: just forward the embedding vector of the kb symbol
  else

    local simple = nn.Sequential()
    simple:add(emb_data)
    simple:add(nn.MaskZero(nn.Mean(1,2),2))-- mean across words in the sequence
    property = nn.Bottle(simple, 2,2)
  end

  local value
  local rnn = nn.Sequential()
  rnn:add(emb_data_val)
  local seqlstm = nn.SeqLSTM(opt.word_vec_size, opt.word_vec_size)
  seqlstm.batchfirst = true
  seqlstm.maskzero = true
  rnn:add(seqlstm)
  rnn:add(nn.Select(2,-1))
  value = nn.Bottle(rnn, 2,2)

  component_data:add(property)
  component_data:add(value)

  data_enc:add(component_data)
  data_enc:add(nn.JoinTable(2,2))
  local compose = nn.MaskZero(nn.Linear(opt.word_vec_size*2, opt.word_vec_size),2)
  data_enc:add(nn.Bottle(compose, 2,2))
  return data_enc
end


function dbEncoder.getDataEncoderOneSequenceLSTM_1D(opt, voc_data_size, propLength, dataPropSeqLength, train)

  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local seqlstm
  seqlstm = nn.SeqLSTM(opt.word_vec_size, opt.rnn_size)
  seqlstm.batchfirst = true
  seqlstm.maskzero = true
  local avged_data1 = nn.Sequential()
  avged_data1:add(emb_data)
  avged_data1:add(nn.Reshape(dataPropSeqLength, propLength, opt.word_vec_size))
  avged_data1:add(nn.View(-1,propLength, opt.word_vec_size)) --get rid of the two first batch dimensions
  avged_data1:add(nn.Contiguous())
  avged_data1:add(seqlstm)
  avged_data1:add(nn.Select(2,-1))
  avged_data1:add(nn.View(-1, dataPropSeqLength, opt.rnn_size)) --restore the two first batch dimensions
  return avged_data1
end


function dbEncoder.getDataEncoderOneSequenceLSTM_2D(opt, voc_data_size, train)

  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local enc = nn.Sequential()
  local seqlstm = nn.Sequential()
  local lstm = nn.SeqLSTM(opt.word_vec_size, opt.rnn_size)
  lstm.batchfirst = true
  lstm.maskzero = true
  seqlstm:add(nn.Contiguous())
  seqlstm:add(lstm)
  seqlstm:add(nn.Select(2,-1))
  local bemb = nn.Bottle(emb_data, 2,3)
  local brnn = nn.Bottle(seqlstm, 3,2)

  enc:add(bemb)
  enc:add(brnn)

  return localize(enc)
end

--[[encodes a single given sequence as averaged sum (bag-of-words)
-- fast implementation as a single multisequence vector  and matrix operations]]
function dbEncoder.getDataEncoderOneSequenceBOW_1D(opt, voc_data_size, propLength, dataPropSeqLength, train)
  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local avged_data1 = nn.Sequential()
  avged_data1:add(emb_data)
  avged_data1:add(nn.MaskZero(nn.Reshape(dataPropSeqLength, propLength, opt.word_vec_size, true),2))
  avged_data1:add(nn.MaskZero(nn.Mean(2,3),3))
  return avged_data1
end


--encodes a single given sequence as averaged sum (bag-of-words)
function dbEncoder.getDataEncoderOneSequenceBOW_2D(opt, voc_data_size, train)
  --[[encode input property set]]

  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)

  local simple = nn.Sequential()
  simple:add(emb_data)
  simple:add(nn.MaskZero(nn.Mean(1,2),2))-- mean across words in the sequence
  simple:add(nn.Contiguous())
  simple:add(nn.View(-1,1, opt.word_vec_size))


  local property = nn.Sequential()
  property:add(nn.SplitTable(1,2))

  local mapl = nn.MapTable()
  mapl:add(simple)
  property:add(mapl)
  property:add(nn.JoinTable(2))

  return property
end

--encodes property-name property-value sequences as composed f(property-name, property-value) with linear composition
function dbEncoder.getDataEncoderStruct(opt, voc_data_size, propNameLength, propLength, dataPropSeqLength, train)
  --[[encode input property set]]

  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local emb_data_val = emb_data:clone('weight', 'bias', 'gradWeight', 'gradBias')

  local data_enc = nn.Sequential()
  local component_data = nn.ParallelTable()

  local property
  if opt.propNameKB then
    --TODO: just forward the embedding vector of the kb symbol
  else
    local simple = nn.Sequential()
    simple:add(emb_data)
    simple:add(nn.MaskZero(nn.Mean(1,2),2))-- mean across words in the sequence
    simple:add(nn.MaskZero(nn.Linear(opt.word_vec_size, opt.word_vec_size/2),2))
    property = nn.Bottle(simple, 2,2)
  end


  local value
  local rnn = nn.Sequential()
  rnn:add(emb_data_val)
  local seqlstm = nn.SeqLSTM(opt.word_vec_size, opt.word_vec_size/2)
  seqlstm.batchfirst = true
  seqlstm.maskzero = true
  rnn:add(seqlstm)
  rnn:add(nn.Select(2,-1))
  value = nn.Bottle(rnn, 2,2)

  component_data:add(property)
  component_data:add(value)

  data_enc:add(component_data)
  data_enc:add(nn.JoinTable(2,2))

  return data_enc
end


--[[DEPRECATED]]
--encodes property-name property-value sequences as composed f(property-name, property-value) with linear composition
function dbEncoder.getDataEncoderStruct_SplitBased(opt, voc_data_size, propNameLength, propLength, dataPropSeqLength, train)
  --[[encode input property set]]

  local emb_data = dbEncoder.getEmbeddingMatrix(opt, voc_data_size, train)
  local emb_data_val = emb_data:clone('weight', 'bias', 'gradWeight', 'gradBias')

  local data_enc = nn.Sequential()
  local component_data = nn.ParallelTable()

  local property
  if opt.propNameKB then
    --TODO: just forward the embedding vector of the kb symbol
  else
    property = nn.Sequential()
    property:add(nn.SplitTable(1,2))

    local simple = nn.Sequential()
    simple:add(emb_data)
    simple:add(nn.MaskZero(nn.Mean(1,2),2))-- mean across words in the sequence
    simple:add(nn.MaskZero(nn.Linear(opt.word_vec_size, opt.word_vec_size/2),2))
    simple:add(nn.Contiguous())
    simple:add(nn.View(-1,1, opt.word_vec_size/2))

    local mapl = nn.MapTable()
    mapl:add(simple)
    property:add(mapl)
    property:add(nn.JoinTable(2))
  end

  local value = nn.Sequential()
  value:add(nn.SplitTable(1, 2))

  local rnn = nn.Sequential()
  rnn:add(emb_data_val)
  local seqlstm = nn.SeqLSTM(opt.word_vec_size, opt.word_vec_size/2)
  seqlstm.batchfirst = true
  seqlstm.maskzero = true
  rnn:add(seqlstm)
  rnn:add(nn.Select(2,-1))
  rnn:add(nn.View(-1,1, opt.word_vec_size/2))

  local map=nn.MapTable()
  map:add(rnn)
  value:add(map)
  value:add(nn.JoinTable(2))

  component_data:add(property)
  component_data:add(value)

  data_enc:add(component_data)
  data_enc:add(nn.JoinTable(2,2))

  return data_enc
end


return dbEncoder