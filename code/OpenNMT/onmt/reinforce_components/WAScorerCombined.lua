
require 'bleu'


local WAScorerCombined = torch.class('WAScorerCombined')

function WAScorerCombined:__init(dict, batch_size)
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

function contains(t, e)
  for i = 1,#t do
    if t[i] == e then return true end
  end
  return false
end

function WAScorerCombined:getDynBatch(y, alns, y_pred, reward, reward_mask)
  --[[
  -- DEPRECATED (f-measure on precision and recall either whole document or block)
  -- ]]
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
      if y[{ i, j }] == onmt.Constants.EOS or y[{ i, j }] == 0 then
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
    local visited = {}
    if #alnwords ~= 0 then
      for k1,v1 in pairs(tgtPred) do
        for k2,v2 in pairs(alnwords) do
          if v1==v2 and not contains(visited, v1) then
            inter = inter + 1 --TODO: have two inter, with and w/o repeted terms for both measures
            table.insert(visited, v1)
          end
        end
      end


      --print(string.format("reward %d/%d " , r, #tgtPred), r/#tgtPred )
      --[[
      --if fmeasure then
      local rPrec = #tgtPred ~= 0 and  r/#tgtPred or 0
      local rRecall = #alnwords ~= 0 and  r/#alnwords or 0
      r = (rPrec ~= 0 or rRec ~= 0) and 2 * ((rPrec * rRec) / (rPrec + rRec)) or 0
      --elseif precision then
      --  r = #tgtPred ~= 0 and  r/#tgtPred or 0
      --end
      ]]

      local alnseq = table.concat(alnwords, ' ')
      local rPrec = get_bleu(tgtPred, alnwords, 1)
      local rRecall = #alnwords ~= 0 and  inter/#alnwords or 0
      r = (rPrec ~= 0 or rRecall ~= 0) and 2 * ((rPrec * rRecall) / (rPrec + rRecall)) or 0

      print(alnseq)
      print(trg_sent)
      print("rew: ", r, rPrec, rRecall)

      --[[
      if tgtPred==0 then
        print("EOS pred ", r)
        print(alnseq)
        r = 0.4
      end
]]
      local cntUninstToks = 0
      for k1,v1 in pairs(tgtPred) do
        if v1=="YEAR" or v1=="YEARs" or v1=="NUMERIC" then
          --print("empty token", v1)
          cntUninstToks = cntUninstToks + 1
        end
      end

      if cntUninstToks > 0 then
          r = r / cntUninstToks
      end

      --[[print(trg_sent)
      print(alnseq)
      print(uniBleu)
      print("\n")]]

      if not inspect(r, "r") then
        print(#tgtPred)
        print(y_pred:size(2))
        os.exit()
      end

      table.insert(ori_sents.alnwords, alnwords)

    end

    if r == 0 then r = 1e-5 end
    assert(last_pos ~= -1, 'last_pos must be valid!')
    reward[{ i , last_pos }] = r

    table.insert(ori_sents.tgt, trg_sent)
    table.insert(ori_sents.ref, ref_sent)
  end
  
  return reward, reward_mask, ori_sents
end

