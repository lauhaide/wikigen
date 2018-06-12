
--include '../utils/shortcut.lua'

require 'bleu'

local WAScorerB1 = torch.class('WAScorerB1')

function WAScorerB1:__init(dict, batch_size)
  self.args = {}
  --self.args.batchSize = batch_size
  self.args.dict = dict
end




local function isnan(x) return x ~= x end
local function inspect(x, msg)
  local ok = true
    if isnan(x) == true then
      print(sys.COLORS.red .. "Nan prediction... " .. msg)
      ok = false
    end
    if x == math.huge then
      print(sys.COLORS.red .. "HUGE prediction... " .. msg)
      ok=false
    end
    return ok
end


function WAScorerB1:getDynBatch(y, alns, y_pred, reward, reward_mask, blockSampleStart)
  --[[
  -- Computes Bleu1 unigram precision as reward.
  -- As target can either use the *block* or *document* alignment set.
   ]]
  reward:zero()
  reward_mask:zero()
  
  local ori_sents = { alnwords = {}, tgt = {}, ref = {} }

  local function get_word( wid)
    return self.args.dict.words:lookup(wid)
  end

  local batchSize = y:size(1)
  for i = 1, batchSize do

    -- get target
    local ref = {}
    local alnwords = {}
    for j = 1, y:size(2) do
      if y[{ i, j }] == onmt.Constants.EOS or y[{ i, j }] == 0 or y[{ i, j }] == onmt.Constants.PAD then
        break
      else
        local w = get_word( y[{ i, j }])
        table.insert(ref, w)
        if alns[{ i, j }] == onmt.Constants.ALIGNID then
          table.insert(alnwords, w)
        end
      end
    end

    -- get predict
    local tgtPred = {}
    local last_pos = -1
    local ended = false
    for j = 2, y_pred:size(2) do
      if y_pred[{ i, j }] == onmt.Constants.PAD then
        --this possition does not count even as last
        --this will happend for fully padded sequences
        break
      end
      if y_pred[{ i, j }] == onmt.Constants.EOS then
        last_pos = j
        ended = true
        reward_mask[{ i, j }] = 1
        break
      else
        table.insert(tgtPred, get_word(y_pred[{ i, j }]))
        last_pos = j
        reward_mask[{ i, j }] = 1
      end
    end

    local trg_sent = table.concat(tgtPred, ' ')
    local ref_sent = table.concat(ref, ' ')
    local r, inter = 0, 0
    if #alnwords ~= 0 and last_pos >=2 then
      for k1,v1 in pairs(tgtPred) do
        for k2,v2 in pairs(alnwords) do
          if v1==v2 then inter = inter + 1
          end
        end
      end

      local alnseq = table.concat(alnwords, ' ')
      r = get_bleu(tgtPred, alnwords, 1)

      if ended and #tgtPred==0 and #ref==0 then
        --the sequence starts with the EOS symbol (i.e. the sequence is of the form <end> <blank> <blank> <blank> ...)
        --yes this happends with the block segmentation of the texts, and what should be the reward? I gues if
        --sampling was alowed in this region and the sample coincides with the reference should be ok. Though
        -- not sure if we sould consider this as a special case.
        r = 0.1
      end

      local prevr = r
      for k1,v1 in pairs(tgtPred) do
        if v1=="YEAR" or v1=="YEARs" or v1=="NUMERIC" or v1=="UNK" then
          r = ( r - 0.025 ) > 0 and r - 0.025 or r
        end
      end

      if not inspect(r, "r") then
        print(#tgtPred)
        print(y_pred:size(2))
        os.exit()
      end

      table.insert(ori_sents.alnwords, alnwords)

    end

    if last_pos < blockSampleStart then
      --make the reward for the ith sequence zero
      --the sequence was covered by NLL. prepare for zero grads the RL gradients
      reward_mask[{ i , {} }]:zero()
    else
      if r == 0 then r = 1e-5 end
      assert(last_pos ~= -1, 'last_pos must be valid!')
      reward[{ i , last_pos }] = r
    end
    
    table.insert(ori_sents.tgt, trg_sent)
    table.insert(ori_sents.ref, ref_sent)
  end
  assert(#ori_sents.tgt == batchSize, 'Outputs should have size of the batch')
  
  return reward, reward_mask, ori_sents
end

