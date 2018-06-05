--
-- Created by IntelliJ IDEA.
-- User: laura
-- Date: 24/06/17
-- Time: 13:14
-- To change this template use File | Settings | File Templates.
--

RPADDING = 'right'
LPADDING = 'left'

--[[Formats a score matrix of M x N, a data sequence of length M and a text
-- sequence of lentgh N to be written into a .csv file for visualization and
-- post-processing of evaluation measures.
-- The data sequence will be right-aligned, the text sequence could be right or left aligned.]]
function adjustMatrix2Sequence(pair, propLength, propSeqLength, fragment_scores, idx2word_text, textPadding)

    local data, text = unpack(pair)
    local actual_text_l = text[text:gt(0)]:size()[1]
    local actual_data_l = data:sum(2):gt(0):sum()

    local jinit, jend
    if textPadding=='left' then
        jinit = text:size()[1]- actual_text_l+1
        jend = text:size()[1]
    else
        jinit = 1
        jend = actual_text_l
    end

    --local dataseq = localize(torch.zeros(propSeqLength, propLength))
    local dataseq = localize(torch.zeros(actual_data_l, propLength))
    local heatmap = localize(torch.zeros(actual_text_l*actual_data_l,4))
    local idx = 1
    for i=1 ,actual_data_l  do
          local jseq = 1
          for j=jinit, jend  do
            heatmap[idx][1] = jseq
            heatmap[idx][2] = i
            heatmap[idx][3] = fragment_scores[i][j]
            heatmap[idx][4] = text[j]
            idx = idx + 1
            jseq = jseq + 1
          end
          dataseq[i]:sub(1, data[i]:size()[1]):copy(data[i])
    end

    return heatmap, dataseq
end

--[[given a matrix of scores M x N  shrinks it to the length of the data
-- and text sequences eliminating paddings.]]
function sequenceFragScores(pair, fragment_scores, padding)
    local data, text = unpack(pair)
    local actual_text_l = text[text:gt(0)]:size()[1]
    local actual_data_l = data:sum(2):nonzero():size()[1]
    local subTensor = Tensor(actual_data_l, actual_text_l)
    if padding=='left' then
        return subTensor:copy(fragment_scores:sub(1, actual_data_l, text:size()[1]- actual_text_l+1, text:size()[1]))
    else
        return subTensor:copy(fragment_scores:sub(1, actual_data_l, 1, actual_text_l))
    end
end

--[[Creates alignment input files for the evaluation tool and .csv files for
-- the alignment visualisation tool.
-- Receives as input a table with two fragment score matrices already formatted
-- an the corresponding data and text sequences.]]
function evalAlignment(waMatrices, idx2word_text, resultDir, computeTreshold)
  local outMax = assert(io.open(resultDir .. "wa_max.txt", "w"))
  local outAll = assert(io.open(resultDir .. "wa_all.txt", "w"))
  local alignment, align, curids, dataseq, seqFragments, textseq, val, pos, sentScore, aType
  local thresholdPossible, thresholdSure = 0.4, 0.7
  if computeTreshold then
    print('define thresholds...')
    thresholdPossible, thresholdSure = getThreshold(waMatrices)
  end

  print(string.format("selection threshold P:%f S:%f", thresholdPossible, thresholdSure))
  for k, pair in pairs(waMatrices) do
    curids, alignment, dataseq, seqFragments, sentScore, textseq = unpack(pair)
    --print(string.format("curid=%d  sentence_ID=%d \n",curids[1], curids[2]-1))
    outMax:write(string.format("curid=%d  sentence_ID=%d (score=%f)\n", curids[1], curids[2]-1, sentScore))
    outAll:write(string.format("curid=%d  sentence_ID=%d (score=%f)\n", curids[1], curids[2]-1, sentScore))

    --all word-property alignment
    for h=1, alignment:size()[1] do
      if alignment[h][3] >= thresholdPossible then
        aType = "P"
        if alignment[h][3] >= thresholdSure then
            aType = "S"
        end

        outAll:write("word=" .. alignment[h][1]-1 .. " relation=" .. alignment[h][2]-1 .. " || "
                .. idx2word_text[alignment[h][4]] .. " || "
                .. strPropSeq(dataseq[alignment[h][2]][dataseq[alignment[h][2]]:gt(0)]) .. " || "
                .. aType .. "\n")
      end
    end

    --only max word-property max alignment
    val, pos = torch.max(seqFragments, 1)
    local maxTextLength = val:size()[2]
    if textseq:size()[1] < maxTextLength then maxTextLength = textseq:size()[1]-1 end --if we got scores for padded seq, reduce 1 for <end>
    for h=1, maxTextLength do
        if val[1][h] >= thresholdPossible then
            aType = "P"
            if val[1][h] >= thresholdSure then
                aType = "S"
            end
            outMax:write("word=" .. h-1 .. " relation=" .. pos[1][h]-1 .. " || "
                    .. idx2word_text[textseq[h]] .. " || "
                    .. strPropSeq(dataseq[pos[1][h]][dataseq[pos[1][h]]:gt(0)])  .. " || "
                    .. aType .. "\n")
        end
    end
    write_csv(alignment, dataseq, curids[1], curids[2]-1, resultDir, thresholdPossible, thresholdSure)
  end
  outAll:close()
  outMax:close()
end

--[[Implements simple formula proposed in https://www.researchgate.net/post/Determination_of_threshold_for_cosine_similarity_score]]
function getThreshold(waMatrices)
    local tmp, first, alignment
    local first = true
    for k, pair in pairs(waMatrices) do
       _, alignment, _, _, _, _ = unpack(pair)
       if first then
         tmp = alignment:select(2,3)
         first = false
       else
         tmp = torch.cat(tmp,alignment:select(2,3),1)
       end
    end
    return  torch.mean(tmp) + 0.75 * torch.std(tmp), torch.mean(tmp) + 2 * torch.std(tmp)
end

--[[Writes the .csv file for the visualisation tool.]]
function write_csv(heatmap, dataseq, ex, k, resultDir, thresholdPossible, thresholdSure)
  local out = assert(io.open(string.format(resultDir .. "ex%d.%d",ex,k) .. '_heatmap.csv', "w"))
  splitter = ","
  local ns, aType

  for i=1,heatmap:size()[1] do
      for j=1,heatmap:size()[2] do
          out:write(heatmap[i][j])
          out:write(splitter)
      end
      --write property sequence
      ns = dataseq[heatmap[i][2]][dataseq[heatmap[i][2]]:gt(0)]
      if (ns:nElement()~=0) then
        out:write(strPropSeq(dataseq[heatmap[i][2]][dataseq[heatmap[i][2]]:gt(0)]))
      else
         print("dataseq is zero", heatmap[i][2])
         --print(dataseq)
         os.exit()
      end
      aType = "U"
      if thresholdPossible~=nil and thresholdSure~=nil then
          if heatmap[i][3]> thresholdSure then
            aType = "S"
          elseif heatmap[i][3]> thresholdPossible then
            aType = "P"
          end
      end
      out:write(splitter .. aType)
      out:write("\n")
  end

  out:close()
end

--[[Given a data sequence generates a string version of it with its indices.]]
function strPropSeq(propIdxTensor)
  ret = ""
  for i=1, propIdxTensor:size()[1] do
    if ret == "" then
      ret = tostring(propIdxTensor[i])
    else
      ret = ret .. "#".. tostring(propIdxTensor[i])
    end
  end
  return ret
end

--[[Reads Yawat alignment files (.aln extension).]]
function readYawatEvalSelection(yawatEvalFile)
  local curidEval = { }
  for l in io.lines(yawatEvalFile) do
    local _, curid = l:match("([^,]+)curid=([^,]+)")
    table.insert(curidEval, tonumber(curid))
  end
  return torch.Tensor(curidEval)
end

--[[Given a sequence and a vocabulary, returns the text version of the sequence.]]
function seqToText(dict, seq)
    --[[for debuggin purpose]]
    local text = ""
    for j=1, seq:size()[1] do
        if dict[seq[j]] ~= nill then
            text = text .. dict[seq[j]] .. " "
        elseif seq[j]==0 then
            text = text .. " 0 "
        else
            print(string.format("no dict entry for index %d", seq[j]))
        end
    end
    return text
end

--[[Given a multi-sequence and a vocabulary, returns the text version of the sequences.
-- Each sequence is separated by a new line (\n).]]
function multiseqToText(dict, seq)

    local text = {}
    for i=1, seq:size()[1] do
        table.insert(text, seqToText(dict, seq[i]))
    end
    return table.concat(text, "\n")
end

--[[Reads a given dictionary.]]
function idx2key(file)
    local f = io.open(file, 'r')
    local t = {}
    local cnt = 0
    for line in f:lines() do
        local c = {}
        for w in line:gmatch '([^%s]+)' do
            table.insert(c, w)
        end
        t[tonumber(c[2])] = c[1]
        cnt = cnt +1
    end
    return t, cnt
end

--[[Given 'source input data' / 'target sequence of words' and alignment scores
-- returns the nb of words in the target sequence that score above a threshold
-- w.r.t some property of the input.
-- ]]
function aligns(seqFragments, noneField, threshold, datasing)
    cnt = 0
    val, pos = torch.max(seqFragments, 1)
    for h=1, val:size()[2] do
        if datasing[pos[1][h]]:eq(noneField):sum() == 0 then -- the alignment does not go to NONEFIELD
            if val[1][h] >= threshold then
                cnt = cnt +1
            end
        end
    end
    return cnt, (seqFragments:size()[2]-1) --do not take into account fullstop ?
end

--[[Given 'source input data' / 'target sequence of words' and alignment scores
-- it generates a sequence of the length of target sequence with 0's 1's
-- depending on whether the target word has an alignment score higher than a
-- threshold with some property of the source input. ]]
function alignmentLabels(seqFragments, noneField, threshold, datasing)
    local labels = {}
    local val, pos = torch.max(seqFragments, 1)
    for h=1, val:size()[2] do
        if datasing[pos[1][h]]:eq(noneField):sum() > 0 then -- the alignment goes to NONEFIELD ##fixed bug here 29/08
            table.insert(labels, "0")
        elseif val[1][h] >= threshold then
            table.insert(labels, "1")
        else
            table.insert(labels, "0")
        end
    end
    return table.concat(labels, ' ')
end