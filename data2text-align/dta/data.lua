--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 16/02/17
-- Time: 11:51
-- To change this template use File | Settings | File Templates.
--


local data = torch.class("data")

function data:__init(opt, data_file)

  print("reading file " .. data_file)
  local f = hdf5.open(data_file, 'r')

  self.local_data_props = f:read('data_props'):all()
  self.local_data_vals = f:read('data_vals'):all()
  self.local_data_single = f:read('data_single'):all()
  self.text = f:read('text'):all()
  self.curids = f:read('curids'):all()

  print(self:getBucketCount())

  self.max_textseq_l = f:read('max_textseq_l'):all()[1]
  self.max_dataseq_l = f:read('max_dataseq_l'):all()[1]

  self.max_prop_l = f:read('max_prop_l'):all()[1]
  self.max_prop_namel = f:read('max_prop_namel'):all()[1]
  self.max_prop_valuel = f:read('max_prop_valuel'):all()[1]

  self.textvoc_size = f:read('textvoc_size'):all()[1]
  self.datavoc_size = f:read('datavoc_size'):all()[1]

  self.buckets_l = f:read('buckets_l'):all()

  self.length = f:read('length'):all()[1]

  print(string.format("Sequence lengths in the dataset are: properties_seq=%d, prop_name_seq=%d, prop_value_seq=%d (or single prop length=%d)",
    self.max_dataseq_l, self.max_prop_namel, self.max_prop_valuel, self.max_prop_l))

  self.cachedBatches = {}
  self.nbCachedMiniBatches = 0
  self.cachedEvaluationBatches = {}
  self.nbCachedEvaluationMiniBatches = 0

  self.NoneField = 2
  self.PAD = 0

  self.all = {}

end

function data:compare(a,b)
  return a[1] < b[1]
end

--[[total nb of buckets in which elements are stored]]
function data:getBucketCount()
  if not (self:tablelength(self.text) == self:tablelength(self.local_data_props)) then
    print("WARNING: data inconsistency, different nb of text and data buckets!")
  end
  return self:tablelength(self.text)
end

function data:tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

--[[total nb of elements in the dataset]]
function data:size()
  return self.length
end

--[[returns the bucket at index i]]
function data:getStructBucket(i)
  local dbp = {}
  local text = {}
  for k, v in pairs(self.local_data_props) do
    if tonumber(k)==i then
      table.insert(dbp, localize(v)) -- v is 3D bucket_length x data_seq_length x prop_value_seq_length
    end
  end
  for k, v in pairs(self.local_data_vals) do
    if tonumber(k)==i then
      table.insert(dbp, localize(v))-- v is 3D bucket_length x data_seq_length x prop_value_seq_length
    end
  end
  for k, v in pairs(self.text) do
    if tonumber(k)==i then
      table.insert(text, localize(v))
    end
  end

  local bucket_size = dbp[1]:size()[1]
  local data_seq_length = dbp[1]:size()[2]
  local text_seq_length = text[1]:size()[2]
  local prop_seq_length = dbp[1]:size()[3] --property-name
  local prop_val_seq_length =  dbp[2]:size()[3] --property-value

  return {dbp, text, bucket_size, data_seq_length, prop_seq_length, prop_val_seq_length, text_seq_length}
end

--[[]]
function data:getSingleSequenceBucket(i)
  local dbp = {}
  local text = {}
  for k, v in pairs(self.local_data_single) do
    if tonumber(k)==i then
      table.insert(dbp, localize(v)) -- v is 3D bucket_length x data_seq_length x prop_name_seq_length+prop_value_seq_length
    end
  end
  for k, v in pairs(self.text) do
    if tonumber(k)==i then
      table.insert(text, localize(v))
    end
  end
  local bucket_size = dbp[1]:size()[1]
  local data_seq_length = dbp[1]:size()[2]
  local prop_seq_length = dbp[1]:size()[3]
  local text_seq_length = text[1]:size()[2]
  return {dbp, text, bucket_size, data_seq_length, prop_seq_length, 0, text_seq_length}
end


function data:getCuridsBucket(i)
  local curid = {}
  for k, v in pairs(self.curids) do
    if tonumber(k)==i then
      table.insert(curid, v) -- v is 3D bucket_length x data_seq_length x prop_value_seq_length
    end
  end

  return curid
end


--[[returns tables of tensors of size Batch_Size taking them from a given bucket]]
function data:getStructBatch(t, bk, bid, batch_l)
  local text = bk[2][1]
  local data_prop = bk[1][1]
  local data_vals = bk[1][2]

  local perm_neg = torch.randperm(bk[3]) --will need to take negative examples from the same bucket

  local bdata_prop = Tensor(batch_l, bk[4], bk[5])
  local bdata_prop_neg = Tensor(batch_l, bk[4], bk[5])
  local bdata_vals = Tensor(batch_l, bk[4], bk[6])
  local bdata_vals_neg = Tensor(batch_l, bk[4], bk[6])
  local btext = Tensor(batch_l, bk[7])
  local btext_neg = Tensor(batch_l, bk[7])

  local negDataVector_prop, negDataVector_vals
  local nb_loops
  local cnti=1
  local negidx = 1
  for i =t, batch_l+t-1 do
    bdata_prop[cnti] = data_prop[i]
    bdata_vals[cnti] = data_vals[i]
    btext[cnti] = text[i]

    if (negidx > bk[3]) then --check bucket size
        print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
        perm_neg = torch.randperm(bk[3]) --will need to take negative examples from the same bucket
        negidx =1
    end

    negDataVector_prop = data_prop[perm_neg[negidx]]
    negDataVector_vals = data_vals[perm_neg[negidx]]

    --if property set of the selected random negative example is equal to the current possitive example
    nb_loops = 0
    while (nb_loops<2 and negidx < bk[3]) and
            (negDataVector_prop[negDataVector_prop:gt(0)]:equal( bdata_prop[cnti][bdata_prop[cnti]:gt(0)] ) )
            and
            (negDataVector_vals[negDataVector_vals:gt(0)]:equal( bdata_vals[cnti][bdata_vals[cnti]:gt(0)] ) ) do
        negidx = negidx +1
        negDataVector_prop = data_prop[perm_neg[negidx]]
        negDataVector_vals = data_vals[perm_neg[negidx]]

      if (negidx == bk[3]) then --check bucket size
          print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
          perm_neg = torch.randperm(bk[3]) --will need to take negative examples from the same bucket
          negidx =1
          nb_loops = nb_loops+1
      end
    end
    if (negidx > bk[3]) then
        print("WARNING: not able to find negative samples for all positive examples in the batch")
        return
    end

    bdata_prop_neg[cnti] = data_prop[perm_neg[negidx]]
    bdata_vals_neg[cnti] = data_vals[perm_neg[negidx]]
    btext_neg[cnti] = text[perm_neg[negidx]]

    cnti = cnti +1
    negidx = negidx +1
  end

  if opt.dataEncoder==1 or opt.dataEncoder==7 then
    return {{bdata_prop, bdata_vals}, btext },{{bdata_prop_neg, bdata_vals_neg}, btext_neg }
  --elseif opt.dataEncoder==3 then
  --   -- use only property-name
  --   return {bdata_prop, btext },{bdata_prop_neg, btext_neg }
  else
    print("ERROR: encoding option not provided")
    return
  end
end

--[[returns tables of tensors of size Batch_Size taking them from a given bucket]]
function data:getSingleSequenceStructBatch(t, bk, bid, batch_l)
  local dbp, text, bucket_size, data_seq_length, prop_seq_length, _, text_seq_length = unpack(bk)

  local data = dbp[1]
  local text = text[1]

  local perm_neg = torch.randperm(bucket_size) --will need to take negative examples from the same bucket

  local bdata = Tensor(batch_l, data_seq_length, prop_seq_length)
  local bdata_neg = Tensor(batch_l, data_seq_length, prop_seq_length)
  local btext = Tensor(batch_l, text_seq_length)
  local btext_neg = Tensor(batch_l, text_seq_length)
  local nb_loops
  local cnti=1
  local negidx = 1
  for i =t, batch_l+t-1 do
    bdata[cnti] = data[i]
    btext[cnti] = text[i]

    if (negidx > bucket_size) then --check bucket size
        print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
        perm_neg = torch.randperm(bucket_size) --will need to take negative examples from the same bucket
        negidx =1
    end

    local negDataVector = data[perm_neg[negidx]]

    --if property set of the selected random negative example is equal to the current possitive example
    nb_loops = 0
    while (nb_loops<2 and negidx < bucket_size) and
            (negDataVector[negDataVector:gt(0)]:equal( bdata[cnti][bdata[cnti]:gt(0)] ) ) do
        negidx = negidx +1
        negDataVector = data[perm_neg[negidx]]

      if (negidx == bucket_size) then --check bucket size
          print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
          perm_neg = torch.randperm(bucket_size) --will need to take negative examples from the same bucket
          negidx =1
          nb_loops = nb_loops+1
      end
    end
    if (negidx > bucket_size) then
        print("WARNING: not able to find negative samples for all positive examples in the batch")
        return
    end

    bdata_neg[cnti] = data[perm_neg[negidx]]
    btext_neg[cnti] = text[perm_neg[negidx]]

    cnti = cnti +1
    negidx = negidx +1
  end

  return {bdata, btext },{bdata_neg, btext_neg }
end

--[[returns tables of tensors of size Batch_Size taking them from a given bucket]]
-- input is hierarchical (2D ) multisequence representation,
-- output is flattened (1D ) multisequence representation
function data:getSingleSequenceFlattenedBatch(t, bk, bid, batch_l)
  local dbp, bktext, bucket_size, data_seq_length, prop_seq_length, _, text_seq_length = unpack(bk)

  local max_seq_len
  if opt.dataEncoder==3 then
    max_seq_len = self.max_prop_namel
  else
    max_seq_len = self.max_prop_l
  end

  local data = dbp[1]
  local text = bktext[1]

  local perm_neg = torch.randperm(bucket_size) --will need to take negative examples from the same bucket

  local bdata = localize(torch.zeros(batch_l, self.max_dataseq_l, max_seq_len))
  local bdata_neg = localize(torch.zeros(batch_l, self.max_dataseq_l, max_seq_len))
  local btext = Tensor(batch_l, text_seq_length)
  local btext_neg = Tensor(batch_l, text_seq_length)
  local nb_loops
  local cnti=1
  local negidx = 1
  local fromIdx
  if data:dim()==3 then
    fromIdx = max_seq_len - data:size()[3] + 1
  end
  for i =t, batch_l+t-1 do
    --bdata[cnti]:sub(1, data[i]:size()[1], 1, data[i]:size()[2]):copy(data[i]) --when right padding
    bdata[cnti]:sub(1, data[i]:size()[1], fromIdx, max_seq_len):copy(data[i]) --when left padding
    btext[cnti] = text[i]

    if (negidx > bucket_size) then --check bucket size
        print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
        perm_neg = torch.randperm(bucket_size) --will need to take negative examples from the same bucket
        negidx =1
    end

    local negDataVector = data[perm_neg[negidx]]

    --if property set of the selected random negative example is equal to the current possitive example
    nb_loops = 0
    while (nb_loops<2 and negidx < bucket_size) and
            (negDataVector[negDataVector:gt(0)]:equal( bdata[cnti][bdata[cnti]:gt(0)] ) ) do
        negidx = negidx +1
        negDataVector = data[perm_neg[negidx]]

      if (negidx == bucket_size) then --check bucket size
          print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
          perm_neg = torch.randperm(bucket_size) --will need to take negative examples from the same bucket
          negidx =1
          nb_loops = nb_loops+1
      end
    end
    if (negidx > bucket_size) then
        print("WARNING: not able to find negative samples for all positive examples in the batch")
        return
    end

    --bdata_neg[cnti]:sub(1, data[i]:size()[1], 1, data[i]:size()[2]):copy(data[perm_neg[negidx]]) --when right padding
    bdata_neg[cnti]:sub(1, data[i]:size()[1], fromIdx, max_seq_len):copy(data[perm_neg[negidx]]) --when left padding
    btext_neg[cnti] = text[perm_neg[negidx]]

    cnti = cnti +1
    negidx = negidx +1
  end

  return {bdata:reshape(batch_l,self.max_dataseq_l*max_seq_len), btext },{bdata_neg:reshape(batch_l,self.max_dataseq_l*max_seq_len), btext_neg }
end

-- TODO: re-factor these variant selection through the class
function data:getBucket(b)
  if opt.dataEncoder==1 or opt.dataEncoder==3 or opt.dataEncoder==7 then
    --"structured property set encoding (hierarchical, i.e. 2D)"
    return self:getStructBucket(b)
  elseif opt.dataEncoder==2 or opt.dataEncoder==4 or opt.dataEncoder==5 or opt.dataEncoder==6  or opt.dataEncoder==0 or opt.dataEncoder==8 then
    --"single sequence property set encoding (hierarchical, i.e. 2D)"
    return self:getSingleSequenceBucket(b)
  end
end

function data:getBatchNS(t, bucket_b, b, batch_l)
  if opt.dataEncoder==1 or opt.dataEncoder==7 then
    --"structured property set encoding (hierarchical, i.e. 2D) batch"
    return self:getStructBatch(t, bucket_b, b, batch_l)
  elseif opt.dataEncoder==2 or opt.dataEncoder==5 or opt.dataEncoder==8 then
    --"single sequence property set encoding (hierarchical, i.e. 2D) batch"
    return self:getSingleSequenceStructBatch(t, bucket_b, b, batch_l)
  elseif opt.dataEncoder==4 or opt.dataEncoder==6 or opt.dataEncoder==3 then
    -- " *falttened* multi-sequence encoding..."
    return self:getSingleSequenceFlattenedBatch(t, bucket_b, b, batch_l)
  end
end

--[[If not already generated, creates a set of minibatches of size Batch_Size and stores them]]
function data:cacheNSMiniBatches()

  if #self.cachedBatches ~= 0 then
    return self.cachedBatches
  end

  print("generating training minibatches...")
  local cntbatch = 0
  local batch, batch_neg, batch_l, bucket_b
  local bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l
  for b=1,self:getBucketCount() do
    bucket_b = self:getBucket(b)
    bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l = unpack(bucket_b)
    print(string.format("Batches from bucket %d with %d elements",b,bsize))
    if bsize> 15 then
        batch_l = opt.max_batch_l
        for t=1, bsize, opt.max_batch_l do
          if bsize - t + 1 < opt.max_batch_l then
            if opt.min_batch_l < bsize - t + 1 then
              print(string.format("Less than a batch (%d)",bsize - t + 1))
              batch_l = bsize - t + 1
            else
                print("Skip...")
                break
            end
          end --not enough to get a batch
          batch, batch_neg = self:getBatchNS(t, bucket_b, b, batch_l)
          table.insert(self.cachedBatches, {batch, batch_neg})
          cntbatch = cntbatch + 1
        end
    else
        print("WARNING: Skip bucket is too small.")
    end
  end
  self.nbCachedMiniBatches = cntbatch
  print(string.format("nb of generated minibatches is %d ", cntbatch))
  return self.cachedBatches
end

--[[informs the nb of current exiting minibatches]]
function data:cachedMBatchesCount()
  return self.nbCachedMiniBatches
end
function data:cachedEvaluationMBatchesCount()
  return self.nbCachedEvaluationMiniBatches
end

--DEPRECATED
--[[merges all elements of all buckets into a single tensor, it pads all data sequences to the maximum lengths]]
function data:bucketToAll()
  local allText = localize(torch.zeros(self.length, self.max_textseq_l))
  local allDataProp = localize(torch.zeros(self.length, self.max_dataseq_l, self.max_prop_namel))
  local allDataVals = localize(torch.zeros(self.length, self.max_dataseq_l, self.max_prop_valuel))

  local bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l
  local istart, iend
  local allcnt = 1
  local bucket_b, dataProp, dataVal
  for b=1,self:getBucketCount() do
    bucket_b = self:getBucket(b)
    bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l = unpack(bucket_b)

    istart =  allcnt
    iend = allcnt + bsize -1
    dataProp, dataVal = unpack(bdata)
    allDataVals[{{istart,iend}, {1, bdata_l},{1, bdata_valseq_l}}] = dataVal

    allDataProp[{{istart,iend}, {1, bdata_l}, {1, bdata_propseq_l}}] = dataProp
    allText[{{istart,iend}, {1, btext_l}}] = btext[1]

    allcnt = iend + 1
  end

  self.all = {{allDataProp, allDataVals}, allText}
end

--[[gets random negative sentences from same biography dataset]]
--randomly picks up distractor sentences which are different from the current example
function data:getNegativeSamplesInBioAll(opt, setLength, nbNegComp)
  --[[Get negative training pairs for evaluation]]
  if opt.genNegValid then
    --[[Generate new pairs]]
    print("generating new pairs of negative examples for the evaluation")
    local distrIDXs =  torch.zeros(setLength, nbNegComp)

    for i =1, setLength do
        local perm_neg = torch.randperm(self.length)
        local negidx = 1
        local posExample_prop = self.all[1][1][i]
        local posExample_vals = self.all[1][2][i]
        local negDataVector_prop, negDataVector_vals
        for j=1, nbNegComp do
          negDataVector_prop = self.all[1][1][perm_neg[negidx]]
          negDataVector_vals = self.all[1][2][perm_neg[negidx]]

          while (negidx < self.length) and (negDataVector_prop[negDataVector_prop:gt(0)]:equal( posExample_prop[posExample_prop:gt(0)] ) )
            and
            ((negDataVector_vals[negDataVector_vals:gt(0)]:equal( posExample_vals[posExample_vals:gt(0)] ) )) do
              negidx = negidx +1
              negDataVector_prop = self.all[1][1][perm_neg[negidx]]
              negDataVector_vals = self.all[1][2][perm_neg[negidx]]
          end
          distrIDXs[i][j] = perm_neg[negidx]
          negidx = negidx +1
        end
    end
    print("done.")
    torch.save(opt.distractorSet, distrIDXs)
    return distrIDXs
  else
    --[[Load existing negative pairs]]
    return torch.load(opt.distractorSet)
  end
end

--[[gets random negative sentences from same biography dataset]]
--randomly picks up distractor sentences which are different from the current example
function data:getNegativeSamplesInBioBuckets(opt, nbNegComp)
  --[[Get negative training pairs for evaluation]]
  local negs = {}
  local distrIDXs, perm_neg, posExample, negDataVector, negidx
  if opt.genNegValid then
    --[[Generate new pairs]]
    print("Generating new pairs of negative examples for the evaluation")

    local bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l, bucket_b, nb_loops
    for b=1,self:getBucketCount() do
      bucket_b = self:getSingleSequenceBucket(b) --work with single sequence encoding, order of ex is the same for both
      bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l = unpack(bucket_b)
      print(string.format("distractors for bucket %d with %d elements",b,bsize))
      distrIDXs =  torch.zeros(bsize, nbNegComp)

      for i =1, bsize do
        perm_neg = torch.randperm(bsize)
        negidx = 1
        posExample = bdata[1][i]
        for j=1, nbNegComp do
          --check need to re-init negatives and counter
          if (negidx > bsize) then --check bucket size
            print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
            perm_neg = torch.randperm(bsize) --will need to take negative examples from the same bucket
            negidx = 1
          end
          negDataVector = bdata[1][perm_neg[negidx]]
              nb_loops = 0
              while (nb_loops < 2 and negidx < bsize) and (negDataVector[negDataVector:gt(0)]:equal(posExample[posExample:gt(0)] ) ) do
              negidx = negidx + 1
              negDataVector = bdata[1][perm_neg[negidx]]
              --check need to re-init negatives and counter
              if (negidx == bsize) then --check bucket size
                print("WARNING: not able to find negative samples for all positive examples in the batch. Reshufle idxs and restart")
                perm_neg = torch.randperm(bsize) --will need to take negative examples from the same bucket
                negidx = 1
                nb_loops = nb_loops + 1
              end
          end
          distrIDXs[i][j] = perm_neg[negidx]
          negidx = negidx + 1
        end
      end
      table.insert(negs, distrIDXs)
    end
    print("done.")
    torch.save(opt.distractorSet, negs)
    return negs
  else
    --[[Load existing negative pairs]]
    return torch.load(opt.distractorSet)
  end
end


function data:getStructRankingBatch(t, bk, bid, curids, batch_l, distrIDXs, nbRanking)
  local text = bk[2][1]
  local data_prop = bk[1][1]
  local data_vals = bk[1][2]

  local bdata_prop = Tensor(batch_l * nbRanking, bk[4], bk[5])
  local bdata_vals = Tensor(batch_l * nbRanking, bk[4], bk[6])
  local btext = Tensor(batch_l * nbRanking, bk[7])
  local bcurids = torch.LongTensor(batch_l * nbRanking, 2)

  local cnti=1
  for i =t, batch_l+t-1 do
    bdata_prop[cnti] = data_prop[i]
    bdata_vals[cnti] = data_vals[i]
    btext[cnti] = text[i]
    bcurids[cnti] = curids[1][i]
    cnti = cnti +1
    if nbRanking > 1 and distrIDXs~=nil then
      for j=1, distrIDXs:size()[2] do
        bdata_prop[cnti] = data_prop[i]
        bdata_vals[cnti] = data_vals[i]
        bcurids[cnti] = curids[1][i]
        btext[cnti] = text[distrIDXs[i][j]]--corrupt text part only
        cnti = cnti +1
      end
    end
  end

  if opt.dataEncoder==1 or opt.dataEncoder==7 then
    return {{bdata_prop, bdata_vals}, btext, bcurids }
  --elseif opt.dataEncoder==3 then
  --   -- use only property-name
  --   return {bdata_prop, btext, bcurids }
  else
    print("ERROR: encoding option not provided")
    return
  end
end

function data:getSingleSequenceRankingBatch(t, bk, bid, curids, batch_l, distrIDXs, nbRanking)
  local dbp, text, bucket_size, data_seq_length, prop_seq_length, _, text_seq_length = unpack(bk)

  local data = dbp[1]
  local text = text[1]
  local bdata = Tensor(batch_l * nbRanking, data_seq_length, prop_seq_length)
  local btext = Tensor(batch_l * nbRanking, text_seq_length)
  local bcurids = torch.LongTensor(batch_l * nbRanking, 2)
  local cnti=1
  for i =t, batch_l+t-1 do
    bdata[cnti] = data[i]
    btext[cnti] = text[i]
    bcurids[cnti] = curids[1][i]
    cnti = cnti +1
    if nbRanking > 1 and distrIDXs~=nil then
      for j=1, distrIDXs:size()[2] do
        bdata[cnti] = data[i]
        bcurids[cnti] = curids[1][i]
        btext[cnti] = text[distrIDXs[i][j]]--corrupt text part only
        cnti = cnti +1
      end
    end
  end

  return {bdata, btext, bcurids}
end

function data:getSingleSequenceFlattenedRankingBatch(t, bk, bid, curids, batch_l, distrIDXs, nbRanking)
  local dbp, text, bucket_size, data_seq_length, prop_seq_length, _, text_seq_length = unpack(bk)

    local max_seq_len
  if opt.dataEncoder==3 then
    max_seq_len = self.max_prop_namel
  else
    max_seq_len = self.max_prop_l
  end

  local data = dbp[1]
  local text = text[1]
  local bdata = localize(torch.zeros(batch_l * nbRanking, self.max_dataseq_l, max_seq_len))
  local btext = Tensor(batch_l * nbRanking, text_seq_length)
  local bcurids = torch.LongTensor(batch_l * nbRanking, 2)
  local cnti=1
  for i =t, batch_l+t-1 do
    bdata[cnti]:sub(1, data[i]:size()[1], 1, data[i]:size()[2]):copy(data[i])
    btext[cnti] = text[i]
    bcurids[cnti] = curids[1][i] --is a tensor contained in a table
    cnti = cnti +1
    if nbRanking > 1 and distrIDXs~=nil then
      for j=1, distrIDXs:size()[2] do
        bcurids[cnti] = curids[1][i]
        bdata[cnti]:sub(1, data[i]:size()[1], 1, data[i]:size()[2]):copy(data[i])
        btext[cnti] = text[distrIDXs[i][j]]--corrupt text part only
        cnti = cnti +1
      end
    end
  end

  return {bdata:reshape(batch_l * nbRanking, self.max_dataseq_l*max_seq_len), btext, bcurids}
end

function data:getBatchRanking(t, bucket_b, b, curid_b, batch_l, distrIDXs, nbRanking)
  if opt.dataEncoder==1 or opt.dataEncoder==7 then
    --"structured property encoding..."
    return self:getStructRankingBatch(t, bucket_b, b, curid_b, batch_l, distrIDXs, nbRanking)
  elseif opt.dataEncoder==2 or opt.dataEncoder==5 or opt.dataEncoder==0 or opt.dataEncoder==8 then
    --"triple BOW encoding..."
    return self:getSingleSequenceRankingBatch(t, bucket_b, b, curid_b, batch_l, distrIDXs, nbRanking)
  elseif opt.dataEncoder==4 or opt.dataEncoder==6 or opt.dataEncoder==3 then
    -- "triple BOW *falttened* encoding..."
    return self:getSingleSequenceFlattenedRankingBatch(t, bucket_b, b, curid_b, batch_l, distrIDXs, nbRanking)
  end
end


--[[if nRanking =0 then just return test pair for simple alignment evaluation, else return test pair and
-- a set of nRanking negatives for ranking-based evaluation.
-- **nbRanking should be multiple of opt.max_batch_l**]]
function data:getEvaluationMiniBatches(nbRanking, distrIDXs)
  print("generating evaluation minibatches...")

  local cntbatch = 0
  local batch, batch_neg, batch_l, bucket_b, curid_b
  local bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l
  for b=1,self:getBucketCount() do
    bucket_b = self:getBucket(b)
    curid_b = self:getCuridsBucket(b)
    bdata, btext, bsize, bdata_l, bdata_propseq_l, bdata_valseq_l, btext_l = unpack(bucket_b)
    print(string.format("Batches from bucket %d with %d elements",b,bsize))
    --if bsize> 15 then
        if nbRanking == 1 then
          batch_l = opt.max_batch_l
        else
          batch_l = opt.max_batch_l / nbRanking
        end
        print(string.format("creating evaluation batches for %d positive examples", batch_l))
        for t=1, bsize, batch_l do
          if bsize - t + 1 < batch_l then
              print(string.format("Less than a batch (%d)",bsize - t + 1))
              batch_l = bsize - t + 1 --in evaluation mode we do not skip small batches
          end
          if nbRanking == 1 then
          --  --reuse existing batch splitting functions, just discard the negative pairs.
          --  batch, batch_neg = self:getBatch(t, bucket_b, b, batch_l)
          --  table.insert(self.cachedEvaluationBatches, {batch})
            batch = self:getBatchRanking(t, bucket_b, b, curid_b, batch_l, distrIDXs, nbRanking)

          else
            batch = self:getBatchRanking(t, bucket_b, b, curid_b, batch_l, distrIDXs[b], nbRanking)
          end
          table.insert(self.cachedEvaluationBatches, {batch})
          cntbatch = cntbatch + 1
        end
    --else
    --    print("WARNING: Skip bucket is too small.")
    --end
  end
  self.nbCachedEvaluationMiniBatches = cntbatch
  print(string.format("nb of generated minibatches is %d ", cntbatch))
  return self.cachedEvaluationBatches
end

function data:getSingleDataSequence(data, j)

  if opt.dataEncoder==1 or opt.dataEncoder==7 then
    --"structured property encoding..."
    --concat both sequences
    local prop_name, prop_value = unpack(data)
    local data_seq_l = prop_name[j]:size()[1]
    local singleseq = localize(torch.zeros(data_seq_l, self.max_prop_namel + self.max_prop_valuel))
    local ns
    for k=1, data_seq_l do
      --if k <= prop_name[j]:size()[1] then
        ns = torch.cat(prop_name[j][k][prop_name[j][k]:gt(0)], prop_value[j][k][prop_value[j][k]:gt(0)])
        if (ns:nElement()~=0) then
          singleseq:sub(k,k,1,ns:size()[1]):copy(ns) --copy sequence right-aligned
        end

      --end
    end
    return singleseq

  elseif opt.dataEncoder==2 or opt.dataEncoder==5 or opt.dataEncoder==0 or opt.dataEncoder==8 then
    --"triple single sequence already 2D representation "
    return data[j]

  elseif opt.dataEncoder==3 then
    --"triple single sequence already flattened representation "
    return data[j]:reshape(self.max_dataseq_l, self.max_prop_namel)

  elseif opt.dataEncoder==4 or opt.dataEncoder==6 or opt.dataEncoder==3 then
    --"triple single sequence already flattened representation "
    return data[j]:reshape(self.max_dataseq_l, self.max_prop_namel + self.max_prop_valuel)

  end

end

--[[auxiliary function to contatenate sequences at same possition of two two tensors
-- eliminating intermediate padding. The two input tensors are 3D :
-- batch_size x data_seq_length x sequence_length1 and batch_size x data_seq_length x sequence_length2
-- returns a single 3D tensor of size: batch_size x data_seq_length x (sequence_length1 + sequence_length2
-- it is working but too slow!!!)]]
function data:concatSequence_DEPRECATED(bdata_prop, bdata_vals)
  local batchlength = bdata_prop:size()[1]
  local dataPropSeq = bdata_prop:size()[2]
  local propNameSeq = bdata_prop:size()[3]
  local propValueSeq = bdata_vals:size()[3]
  local allbdata_tmp = localize(torch.zeros(batchlength, dataPropSeq, propNameSeq+propValueSeq ))
  --for batchi=1, batchlength do
  --  allbdata_tmp[{{batchi,batchi}, {1,dataPropSeq},{1,propNameSeq}}] = bdata_prop[batchi]
  --  allbdata_tmp[{{batchi,batchi}, {1,dataPropSeq},{propNameSeq+1,propNameSeq+propValueSeq}}] = bdata_vals[batchi]
  --end
  local allbdata = localize(torch.zeros(batchlength, dataPropSeq, propNameSeq+propValueSeq ))
  local nonzero_seq_prop, nonzero_seq_val
  for batchi=1, batchlength do
    for propi=1, dataPropSeq do
      nonzeros_prop = bdata_prop[batchi][propi]:sum()
      --nonzeros_val = bdata_vals[batchi][propi]:sum()
      if (nonzeros_prop~=0 ) then
        nz_len = torch.nonzero(bdata_prop[batchi][propi]):size()[1]
        newseq = torch.cat(bdata_prop[batchi][propi][{{1,nz_len}}], bdata_vals[batchi][propi])
        allbdata[batchi][propi][{{1,nz_len+bdata_vals[batchi][propi]:size()[1]}}] = newseq

      end
    end
  end
  return allbdata
  --return allbdata_tmp
end


return data




