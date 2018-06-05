--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 09/03/17
-- Time: 15:46
-- To change this template use File | Settings | File Templates.
--
--require 'cudnn'
--require 'dpnn'
--require 'dta.Unsqueeze_nc'

--require 'dta.Bottle_My'
local dbEncoder  = require('dta.dbEncoder')
local embeddingl = require('dta.embeddingl')

function getSentenceEncoder(opt, voc_text_size, train)
  --[[encode input text]]

  local emb_text
  if not train then
    --just create the layer, then weights will be orver-writen by the load model function.
    emb_text = nn.LookupTableMaskZero(voc_text_size, opt.word_vec_size)
  else
    if opt.preembed then
      --load pre-trained word vectors
      emb_text = embeddingl.initPreTrainedEmbeddingMatrix(voc_text_size, opt.word_vec_size, opt.preembed_textvoc)
    else
      emb_text = embeddingl.initRandomEmbeddingMatrix(voc_text_size, opt.word_vec_size)
    end
  end

  local brnn_text1  = nn.Sequential()
  brnn_text1:add(emb_text)
  --brnn_text1:add(nn.Dropout())
  local rnn_l1 = nn.SeqBRNN(opt.word_vec_size, opt.rnn_size/2, true, nn.JoinTable(2,2))
  rnn_l1.maskzero = true
  brnn_text1:add(rnn_l1) --text encoder text
  if opt.num_layers == 2 then --this is not really two layers but 2 stacked LSTMs
    local rnn_l2 = nn.SeqBRNN(opt.rnn_size, opt.rnn_size, true)
    rnn_l2.maskzero = true
    brnn_text1:add(rnn_l2) --text encoder text
  end

  return brnn_text1
end


function train_alignment_scorer(opt, voc_data_size, voc_text_size, dataset)

  local dbp = nn.Identity()()
  local dbp_neg = nn.Identity()()

  local avged_data1_l = dbEncoder.build(opt, dataset, voc_data_size, true)
  local avged_data2 = avged_data1_l:clone('weight', 'bias', 'gradWeight', 'gradBias')
  local enc_data1 = avged_data1_l(dbp)
  local enc_data2 = avged_data2(dbp_neg)

  local brnn_text1_l = getSentenceEncoder(opt, voc_text_size, true)
  local brnn_text2 = brnn_text1_l:clone('weight', 'bias', 'gradWeight', 'gradBias')

  local sent = nn.Identity()()
  local sent_neg = nn.Identity()()
  local enc_text1 = brnn_text1_l(sent)
  local enc_text2 = brnn_text2(sent_neg)

  if opt.selfAttn then
    os.exit()
    local replicateIn1 = nn.ConcatTable()
                      :add(nn.Identity())
                      :add(nn.Identity())(enc_text1)
    local selfDot1 = nn.MM(false, true)(replicateIn1) --transpose the 2nd
    local attnScores1 = nn.Sequential()
        :add(nn.Bottle(nn.Normalize(2), 2))
        :add(nn.Bottle(nn.SoftMax(), 2))
    local selfAttScores1 = attnScores1(selfDot1)
    local attnOutput1 = nn.MM(false, false)({selfAttScores1, enc_text1})
    enc_text1 = nn.CAddTable(2)({attnOutput1, enc_text1})

    --repeat for distractor text for now then re-factor!!!
    --TODO: this should be clonned !!!! Though not so serious as there are no parameters on these layers
    local replicateIn2 = nn.ConcatTable()
                      :add(nn.Identity())
                      :add(nn.Identity())(enc_text2)
    local selfDot2 = nn.MM(false, true)(replicateIn2) --transpose the 2nd
    local attnScores2 = nn.Sequential()
        :add(nn.Bottle(nn.Normalize(2), 2))
        :add(nn.Bottle(nn.SoftMax(), 2))
    local selfAttScores2 = attnScores2(selfDot2)
    local attnOutput2 = nn.MM(false, false)({selfAttScores2, enc_text2})
    enc_text2 = nn.CAddTable(2)({attnOutput2, enc_text2})
  end


  local score_ds_mat = nn.MaskZero(nn.MM(false, true),2)({enc_data1, enc_text1})
  local score_dsneg_mat = nn.MaskZero(nn.MM(false, true),2)({enc_data1, enc_text2})
  local score_dnegs_mat = nn.MaskZero(nn.MM(false, true),2)({enc_data2, enc_text1})

  local s1 = nn.Sequential():add(nn.MaskZero(nn.Max(1, 2),2)):add(nn.MaskZero(nn.Sum(2),1))
  local s2 = nn.Sequential():add(nn.MaskZero(nn.Max(1, 2),2)):add(nn.MaskZero(nn.Sum(2),1))
  local s3 = nn.Sequential():add(nn.MaskZero(nn.Max(1, 2),2)):add(nn.MaskZero(nn.Sum(2),1))

  local score_ds = s1(score_ds_mat)
  local score_dsneg = s2(score_dsneg_mat)
  local score_dnegs = s3(score_dnegs_mat)

  local p1 = nn.ParallelTable():add(nn.Identity()):add(nn.Identity())
  local p2 = nn.ParallelTable():add(nn.Identity()):add(nn.Identity())

  local rankText = p1({score_ds, score_dsneg})
  local rankData = p2({score_ds, score_dnegs})

  return localize(nn.gModule({dbp, sent, dbp_neg, sent_neg},{rankText, rankData})), brnn_text1_l

end


function load_alignment_scorer(opt, voc_data_size, voc_text_size, dataset, weightsFile)
  local dbp = nn.Identity()()

  local avged_data1_l = dbEncoder.build(opt, dataset, voc_data_size, false)
  local enc_data1 = avged_data1_l(dbp):annotate{name = 'DataEncoder', description = 'knowledge base encoder'}

  local brnn_text1_l = getSentenceEncoder(opt, voc_text_size, false)
  local sent = nn.Identity()()
  local enc_text1 = brnn_text1_l(sent):annotate{name = 'TextEncoder', description = 'text encoder'}

  if opt.selfAttn then
    local replicateIn1 = nn.ConcatTable()
                      :add(nn.Identity())
                      :add(nn.Identity())(enc_text1)
    local selfDot1 = nn.MM(false, true)(replicateIn1) --transpose the 2nd
    local attnScores1 = nn.Sequential()
        :add(nn.Bottle(nn.Normalize(2), 2))
        :add(nn.Bottle(nn.SoftMax(), 2))
    local selfAttScores1 = attnScores1(selfDot1)
    local attnOutput1 = nn.MM(false, false)({selfAttScores1, enc_text1})
    enc_text1 = nn.CAddTable(2)({attnOutput1, enc_text1})
  end

  local score_ds_mat = nn.MaskZero(nn.MM(false, true),2)({enc_data1, enc_text1})

  local s1 = nn.Sequential():add(nn.MaskZero(nn.Max(1, 2), 2)):add(nn.MaskZero(nn.Sum(2),1))

  local score_ds = s1(score_ds_mat)

  local fragsorer = localize(nn.gModule({dbp, sent},{enc_data1, enc_text1, score_ds, score_ds_mat}))

  load_parameters(fragsorer, weightsFile)

  return localize(fragsorer)
end


function getCriterion(opt)
    local criterion = nn.ParallelCriterion(false) --multictriterion for Fragment and Global objectives
    criterion:add(nn.MaskZeroCriterion(nn.MarginRankingCriterion(1),0))  --margin ranking sentence
    criterion:add(nn.MaskZeroCriterion(nn.MarginRankingCriterion(1),0))  --margin ranking data
  return localize(criterion)
end


function train_batch(scorer, criterion, batch, batch_neg, optim_func, optim_state)
  local data, text = unpack(batch)
  local data_neg, text_neg = unpack(batch_neg)
  --local targetGlobal = localize(torch.ones(opt.max_batch_l))
  local targetGlobal = localize(torch.ones(text:size()[1]))

  local x, dl_dx = scorer:getParameters()

  local feval = function(x_new)
      -- set x to x_new, if different
      -- (in this simple example, x_new will typically always point to x,
      -- so this copy is never happening)
      if x ~= x_new then
          x:copy(x_new)
      end

      dl_dx:zero()

      -- evaluate the loss function and its derivative wrt x, for that sample
      local criterion_scores = scorer:forward({data, text, data_neg, text_neg})

      local loss_x = criterion:forward(criterion_scores, {targetGlobal, targetGlobal})

      --io.stdout:write('Loss: '..loss_x.."\n" )
      scorer:backward({data, text, data_neg, text_neg}, criterion:backward(criterion_scores, {targetGlobal, targetGlobal}))

      return loss_x, dl_dx
  end

  local optim_x, ls = optim_func(feval, x, optim_state)

  return ls[1] --return batch loss
end


