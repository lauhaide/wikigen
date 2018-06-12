
--include '../utils/shortcut.lua'

require 'bleu'

local WAScorerCombined2 = torch.class('WAScorerCombined2')

function WAScorerCombined2:__init(dict, batch_size)
  self.args = {}
  --self.args.batchSize = batch_size
  self.args.dict = dict
  self.reward2 = localize(torch.Tensor())
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

function contains(t, e)
  for i = 1,#t do
    if t[i] == e then return true end
  end
  return false
end

function WAScorerCombined2:getDynBatch(yLocal, alnsLocal, yGlobal, alnsGlobal, y_pred, reward, reward_mask, blockSampleStart)
  --[[
  -- Computes Bleu1 unigram precision against the *block* alignment set and recall against the *document*
  -- alignment set. Returns both numbers as separated rewards.
   ]]
  reward:zero()
  reward_mask:zero()

  --output recall as separated reward
  self.reward2:resizeAs(reward):zero()
  
  local ori_sents = { alnwords = {}, tgt = {}, ref = {} }

  local function get_word( wid)
    return self.args.dict.words:lookup(wid)
  end

  local batchSize = yLocal:size(1)
  for i = 1, batchSize do

    -- get target and alignment set for block level precision
    local ref = {}
    local alnwordsLocal = {}
    for j = 1, yLocal:size(2) do
      if yLocal[{ i, j }] == onmt.Constants.EOS or yLocal[{ i, j }] == onmt.Constants.PAD then
        break
      else
        local w = get_word( yLocal[{ i, j }])
        table.insert(ref, w)
        if alnsLocal[{ i, j }] == onmt.Constants.ALIGNID then
          table.insert(alnwordsLocal, w)
        end
      end
    end

    -- get target and alignment set for document level for recall
    --local ref = {}
    local alnwordsGlobal = {}
    for j = 1, yGlobal:size(2) do
      if yGlobal[{ i, j }] == onmt.Constants.EOS or yGlobal[{ i, j }] == onmt.Constants.PAD then
        break
      else
        local w = get_word( yGlobal[{ i, j }])
        --table.insert(ref, w)
        if alnsGlobal[{ i, j }] == onmt.Constants.ALIGNID then
          table.insert(alnwordsGlobal, w)
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
    local inter, rPrec, rRecall = 0, 0, 0
    local visited = {}
    if #alnwordsGlobal ~= 0 and last_pos >=2 then
      for k1,v1 in pairs(tgtPred) do
        for k2,v2 in pairs(alnwordsGlobal) do
          if v1==v2 and not contains(visited, v1) then
            inter = inter + 1 --TODO: have two inter, with and w/o repeted terms for both measures
            table.insert(visited, v1)
          end
        end
      end

      local alnseqBlock = table.concat(alnwordsLocal, ' ')
      local alnseqDoc = table.concat(alnwordsGlobal, ' ')
      rPrec = get_bleu(tgtPred, alnwordsLocal, 1)
      rRecall = #alnwordsGlobal ~= 0 and  inter/#alnwordsGlobal or 0
      --r = (rPrec ~= 0 or rRecall ~= 0) and 2 * ((rPrec * rRecall) / (rPrec + rRecall)) or 0

      if ended and #tgtPred==0 and #ref==0 then
        --the sequence starts with the EOS symbol (i.e. the sequence is of the form <end> <blank> <blank> <blank> ...)
        --yes this happends, and what should be the reward? I gues if sampling was alowed in this region and
        --the sample coincides with the reference should be ok. Though nos sure if we sould consider this as
        --a special case.
        rPrec = 0.1
        rRecall = 0.1
      end


      for k1,v1 in pairs(tgtPred) do
        if v1=="YEAR" or v1=="YEARs" or v1=="NUMERIC" or v1=="UNK"  then
          rPrec = ( rPrec - 0.025 ) > 0 and rPrec - 0.025 or rPrec
        end
      end

      if not inspect(rPrec, "rPrec") or not inspect(rRecall, "rRecall") then
        print(ref_sent)
        print(alnseqBlock)
        print(trg_sent)
        print(#tgtPred)
        print(y_pred:size(2))
        print("rew: ", rPrec, rRecall, last_pos)
        os.exit()
      end

      table.insert(ori_sents.alnwords, alnwordsGlobal)

    end

    if last_pos < blockSampleStart then
      --make the reward for the ith sequence zero
      --the sequence was covered by NLL. prepare for zero grads the RL gradients
      reward_mask[{ i , {} }]:zero()
    else
      if rPrec == 0 then rPrec = 1e-5 end
      if rRecall == 0 then rRecall = 1e-5 end
      assert(last_pos ~= -1, 'last_pos must be valid!')
      reward[{ i , last_pos }] = rPrec
      self.reward2[{ i , last_pos }] = rRecall
    end
--print("rew: ", rPrec, rRecall, last_pos, blockSampleStart, reward_mask[{ i , {} }]:sum())
    table.insert(ori_sents.tgt, trg_sent)
    table.insert(ori_sents.ref, ref_sent)

  end
  assert(#ori_sents.tgt == batchSize, 'Outputs should have size of the batch')

  return reward, self.reward2, reward_mask, ori_sents
end

