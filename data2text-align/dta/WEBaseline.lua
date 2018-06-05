--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 03/05/17
-- Time: 11:29
-- To change this template use File | Settings | File Templates.
--




function getDataEncoderOneSequenceBOW(opt, voc_data_size, propLength, dataPropSeqLength, train)
  --local emb_data = initPreTrainedEmbeddingMatrix(voc_data_size, opt.word_vec_size, opt.preembed_datavoc)
  local emb_data = initPreTrainedEmbeddingMatrixL2(voc_data_size, opt.word_vec_size, opt.preembed_datavoc)
  local avged_data1 = nn.Sequential()
  avged_data1:add(emb_data)
  avged_data1:add(nn.MaskZero(nn.Reshape(dataPropSeqLength, propLength, opt.word_vec_size, true),2))
  avged_data1:add(nn.MaskZero(nn.Mean(2,3),3))
  return avged_data1
end

function getSentenceWordEncoder(opt, voc_text_size)
  --local emb_text = initPreTrainedEmbeddingMatrix(voc_text_size, opt.word_vec_size, opt.preembed_textvoc)
  local emb_text = initPreTrainedEmbeddingMatrixL2(voc_text_size, opt.word_vec_size, opt.preembed_textvoc)
  local textEnc = nn.Sequential()
  textEnc:add(emb_text)
  return textEnc
end

function embBaselineScorer(opt, voc_data_size, voc_text_size)
  local dbp = nn.Identity()()
  local avged_data1_l = getDataEncoderOneSequenceBOW_2D(opt, voc_data_size, false)
  local normDataEnc = nn.Sequential()
  normDataEnc:add(avged_data1_l)
  normDataEnc:add(nn.Bottle(nn.Normalize(2), 2, 2))
  --local enc_data1 = avged_data1_l(dbp)
  local enc_data1 = normDataEnc(dbp)

  local brnn_text1_l = getSentenceWordEncoder(opt, voc_text_size)
  local sent = nn.Identity()()
  local enc_text1 = brnn_text1_l(sent)

  local score_ds_mat = nn.MaskZero(nn.MM(false, true),2)({enc_data1, enc_text1})

  local s1 = nn.Sequential():add(nn.MaskZero(nn.Max(1, 2), 2)):add(nn.MaskZero(nn.Sum(2),1))

  local score_ds = s1(score_ds_mat)

  return localize(nn.gModule({dbp, sent},{enc_data1, enc_text1, score_ds, score_ds_mat}))
end


