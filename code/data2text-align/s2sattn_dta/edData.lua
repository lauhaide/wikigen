--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 16/02/17
-- Time: 11:51
-- To change this template use File | Settings | File Templates.
--


local edData = torch.class("edData")

function edData:__init(opt, data_file)

  print("reading file " .. data_file)
  local f = hdf5.open(data_file, 'r')

  self.local_data_props = f:read('data_props'):all()
  self.local_data_vals = f:read('data_vals'):all()
  self.local_data_single = f:read('data_single'):all()
  self.targetIn = f:read('targetIn'):all()
  self.targetOut = f:read('targetOut'):all()
  self.curids = f:read('curids'):all()
  if opt.guided_alignment==1 or opt.reinforce then
    print(" * Load alignment labels")
    self.alnLabels = f:read('alnlbl'):all()
  end

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
  --self.nbCachedMiniBatches = 0  --- TODO remoce this field, shuld be given on the fly based on table cachedBatches
  self.cachedEvaluationBatches = {}
  self.nbCachedEvaluationMiniBatches = 0 --TODO same, remove
  self.maskNoneField = false
  --read the following two fro; the input hdf5
  self.NoneField = 2
  self.PAD = 0

  self.all = {}

end

function edData:compare(a,b)
  return a[1] < b[1]
end

--[[total nb of buckets in which elements are stored]]
function edData:getBucketCount()
  if not (self:tablelength(self.targetIn) == self:tablelength(self.local_data_props)) then
    print("WARNING: data inconsistency, different nb of text and data buckets!")
  end
  return self:tablelength(self.targetIn)
end

function edData:tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count

  --return #T this would do ???
end

--[[total nb of elements in the dataset]]
function edData:size()
  return self.length
end

--[[returns the bucket at index i
-- ]]
function edData:getStructBucket(i)
  local dbp = {}
  local target = {}
  local bcurid = {}
  local alnLabels = {}
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
  for k, v in pairs(self.targetIn) do
    if tonumber(k)==i then
      table.insert(target, localize(v))
    end
  end
  for k, v in pairs(self.targetOut) do
    if tonumber(k)==i then
      table.insert(target, localize(v))
    end
  end
  for k, v in pairs(self.curids) do
    if tonumber(k)==i then
      -- NOTE: see note on getSingleSequenceBucket()
      table.insert(bcurid, v)
    end
  end
  if opt.guided_alignment==1 or opt.reinforce then
    for k, v in pairs(self.alnLabels) do
      if tonumber(k)==i then
        table.insert(alnLabels, v)
      end
    end
  end

  local bucket_size = dbp[1]:size()[1]
  local data_seq_length = dbp[1]:size()[2]
  local text_seq_length = target[1]:size()[2]
  local prop_seq_length = dbp[1]:size()[3] --property-name
  local prop_val_seq_length =  dbp[2]:size()[3] --property-value

  --TODO: do sgth else?
  --target hierarchical not used with this source data type of encoding, just add the return element =0 for compatibility
  return {dbp, target, bucket_size, data_seq_length, prop_seq_length, prop_val_seq_length, 0, text_seq_length, bcurid, alnLabels}
end

--[[]]
function edData:getSingleSequenceBucket(i)
  local dbp = {}
  local target = {}
  local bcurid = {}
  local alnLabels = {}
  for k, v in pairs(self.local_data_single) do
    if tonumber(k)==i then
      table.insert(dbp, localize(v)) -- v is 3D bucket_length x data_seq_length x prop_name_seq_length+prop_value_seq_length
    end
  end
  for k, v in pairs(self.targetIn) do
    if tonumber(k)==i then
      table.insert(target, localize(v))
    end
  end
  for k, v in pairs(self.targetOut) do
    if tonumber(k)==i then
      table.insert(target, localize(v))
    end
  end
  for k, v in pairs(self.curids) do
    if tonumber(k)==i then
      -- NOTE: do not use localize() here as it was strangely modifying the data (i.e. curids)
      table.insert(bcurid, v)
    end
  end
  if opt.guided_alignment==1 or opt.reinforce then
    for k, v in pairs(self.alnLabels) do
      if tonumber(k)==i then
        table.insert(alnLabels, v)
      end
    end
  end

  local bucket_size = dbp[1]:size()[1]
  local data_seq_length = dbp[1]:size()[2]
  local prop_seq_length = dbp[1]:size()[3]
  local sent_seq_length, doc_length = 0
  if opt.hdec then
    doc_length = target[1]:size()[2]
    sent_seq_length = target[1]:size()[3]
  else
    sent_seq_length = target[1]:size()[2]
  end

  --print(string.format("bucket: %d ; propertySetSize: %d ; prop_seq_length: %d", i, data_seq_length, prop_seq_length))
  return {dbp, target, bucket_size, data_seq_length, prop_seq_length, 0, doc_length, sent_seq_length, bcurid, alnLabels}
end

function edData:maskInputNoneField()
  self.maskNoneField = true
end

function edData:getBucketSize(bk)
  return bk[3]
end

--[[Returns a batch of size batchSize taken from a given bucket.
--  Source data is hirarchical and property-name property-value are separated sequences.
--  Target sequence is single sequence. TODO: prep for hirarchical target?
-- ]]
function edData:getStructBatch(t, bk, bid, batch_l)
  local targetIn = bk[2][1]
  local targetOut = bk[2][2]
  local data_prop = bk[1][1]
  local data_vals = bk[1][2]
  local curids = bk[9]
  local alnLbls = bk[10]

  local bdata_prop = Tensor(batch_l, bk[4], bk[5])
  local bdata_vals = Tensor(batch_l, bk[4], bk[6])
  local btargetIn = Tensor(batch_l, bk[8])
  local btargetOut = Tensor(batch_l, bk[8])
  local balnLbls = Tensor(batch_l, bk[8])
  local bcurid = torch.LongTensor(batch_l, 2)

  local cnti=1
  for i =t, batch_l+t-1 do
    bdata_prop[cnti] = data_prop[i]
    bdata_vals[cnti] = data_vals[i]
    btargetIn[cnti] = targetIn[i]
    btargetOut[cnti] = targetOut[i]
    if opt.guided_alignment==1 or opt.reinforce then
      balnLbls[cnti] = alnLbls[i]
    end
    bcurid[cnti] = curids[1][i]
    cnti = cnti +1
  end

  if opt.dataEncoder==1 or opt.dataEncoder==7 then
    return {{bdata_prop, bdata_vals}, btargetIn, btargetOut, balnLbls, bcurid}
  else
    print("ERROR: encoding option not provided")
    return
  end
end

--[[Returns a batch of size batchSize taken from a given bucket.
-- Each source data unit consists of a set of properties, each property
-- is encoded as a single sequence [property-name ; property-value].
-- Target can be a single sequence (1D) or hierarchical on sentence-document (2D).
-- ]]
function edData:getSingleSequenceStructBatch(t, bk, bid, batch_l, fb)
  local dbp, target, _, data_seq_length, prop_seq_length, _, doc_length, text_seq_length, curids, alnLbls = unpack(bk)

  local data = dbp[1]
  local targetIn = target[1]
  local targetOut = target[2]
  alnLbls = alnLbls[1]

  local bdata = Tensor(batch_l, data_seq_length, prop_seq_length)
  local btargetIn, btargetOut
  if opt.hdec then
    btargetIn = Tensor(batch_l, doc_length, text_seq_length)
    btargetOut = Tensor(batch_l, doc_length, text_seq_length)
  else
    btargetIn = Tensor(batch_l, text_seq_length)
    btargetOut = Tensor(batch_l, text_seq_length)
  end
  local balnLbls = Tensor(batch_l, text_seq_length)
  local bcurid = torch.LongTensor(batch_l, 2)

  local cnti=1
  for i =t, batch_l+t-1 do
    if self.maskNoneField then
      bdata[cnti] = data[i]:maskedFill(data[i]:eq(self.NoneField),self.PAD) --TODO move pad to constants
    else
      bdata[cnti] = data[i]
    end
    btargetIn[cnti] = targetIn[i]
    btargetOut[cnti] = targetOut[i]
    if opt.guided_alignment==1 or opt.reinforce then
      balnLbls[cnti] = alnLbls[i]
    end
    bcurid[cnti] = curids[1][i]
    cnti = cnti +1
  end
  return {bdata, btargetIn, btargetOut, balnLbls, bcurid}
end

--[[Returns a batch of size batchSize taken from a given bucket.
-- source data units are hierarchical (2D ) multisequence representation,
-- but this batch will flatten them (1D ) into a single sequence.
-- Target is single sequence (1D) TODO: adapt for hirarchical target???
--]]
function edData:getSingleSequenceFlattenedBatch(t, bk, bid, batch_l)
  local dbp, bktarget, _, _, _, _, doc_length, text_seq_length, curids, alnLbls = unpack(bk)

  local max_seq_len
  if opt.dataEncoder==3 then
    max_seq_len = self.max_prop_namel
  else
    max_seq_len = self.max_prop_l
  end

  local data = dbp[1]
  local targetIn = bktarget[1]
  local targetOut = bktarget[2]
  alnLbls = alnLbls[1]

  local bdata = localize(torch.zeros(batch_l, self.max_dataseq_l, max_seq_len))
  local btargetIn = Tensor(batch_l, text_seq_length)
  local btargetOut = Tensor(batch_l, text_seq_length)
  local balnLbls = Tensor(batch_l, text_seq_length)
  local bcurid = torch.LongTensor(batch_l, 2)

  local cnti=1
  local fromIdx
  if data:dim()==3 then
    fromIdx = max_seq_len - data:size()[3] + 1
  end
  for i =t, batch_l+t-1 do
    bdata[cnti]:sub(1, data[i]:size()[1], fromIdx, max_seq_len):copy(data[i]) --when left padding
    btargetIn[cnti] = targetIn[i]
    btargetOut[cnti] = targetOut[i]
    if opt.guided_alignment==1 or opt.reinforce then
      balnLbls[cnti] = alnLbls[i]
    end
    bcurid[cnti] = curids[1][i]
    cnti = cnti +1
  end

  return {bdata:reshape(batch_l,self.max_dataseq_l*max_seq_len), btargetIn, btargetOut, balnLbls, bcurid}
end

-- TODO: re-factor these variant selection through the class
function edData:getBucket(b, dataEncoder, htextEncoder)
  if dataEncoder==1 or dataEncoder==3 or dataEncoder==7 then
    --"separated property elements encoding (hierarchical, i.e. 2D)"
    return self:getStructBucket(b)
  elseif dataEncoder==2 or dataEncoder==4 or dataEncoder==5 or dataEncoder==6  or dataEncoder==0 or dataEncoder==8 then
    --"single sequence property elements encoding (hierarchical, i.e. 2D)"
    return self:getSingleSequenceBucket(b)
  end
end

function edData:__getBatch(t, bucket_b, b, batch_l, dataEncoder)
  if dataEncoder==1 or dataEncoder==7 then
    --"structured property set encoding (hierarchical, i.e. 2D) batch"
    return self:getStructBatch(t, bucket_b, b, batch_l)
  elseif dataEncoder==2 or dataEncoder==5 or dataEncoder==8 then
    --"single sequence property set encoding (hierarchical, i.e. 2D) batch"
    return self:getSingleSequenceStructBatch(t, bucket_b, b, batch_l)
  elseif dataEncoder==4 or dataEncoder==6 or dataEncoder==3 then
    -- " *falttened* multi-sequence encoding..."
    return self:getSingleSequenceFlattenedBatch(t, bucket_b, b, batch_l)
  end
end



--[[If not already generated, creates a set of minibatches of size Batch_Size and stores them]]
function edData:cacheMiniBatches(batchSize, dataEncoder, testMode)

  if #self.cachedBatches ~= 0 then
    return self.cachedBatches
  end

  print("generating training minibatches...")
  local cntbatch = 0
  local batch, batch_l, bucket_b
  local  bsize
  for b=1,self:getBucketCount() do
    bucket_b = self:getBucket(b, dataEncoder)
    bsize = self:getBucketSize(bucket_b)
    print(string.format("Batches from bucket %d with %d elements",b,bsize))
    if bsize> 15 then
        batch_l = batchSize --opt.max_batch_l
        for t=1, bsize, batch_l do
          if bsize - t + 1 < batch_l then
            ----if opt.min_batch_l < bsize - t + 1 then
            if testMode and 0 < bsize - t + 1 then
              --if in test mode then also return batches of smaller size
              print(string.format("Less than a batch (%d)",bsize - t + 1))
              batch_l = bsize - t + 1
            else
                print("Skip data... ")
                break
            end
          end --not enough to get a batch
          batch = self:__getBatch(t, bucket_b, b, batch_l, dataEncoder)
          table.insert(self.cachedBatches, {batch})
          cntbatch = cntbatch + 1
        end
    else
        print("WARNING: Skip bucket is too small.")
    end
  end
  --self.nbCachedMiniBatches = cntbatch
  print(string.format("nb of generated minibatches is %d ", cntbatch))
  assert(#self.cachedBatches==cntbatch)
  return self.cachedBatches
end

--[[informs the nb of current exiting minibatches]]
function edData:cachedMBatchesCount()
  return #self.cachedBatches
end



function edData:getSingleDataSequence(data, j, dataEncoder)

  if dataEncoder==1 or dataEncoder==7 then
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

  elseif dataEncoder==2 or dataEncoder==5 or dataEncoder==0 or dataEncoder==8 then
    --"triple single sequence already 2D representation "
    return data[j]

  elseif dataEncoder==3 then
    --"triple single sequence already flattened representation "
    return data[j]:reshape(self.max_dataseq_l, self.max_prop_namel)

  elseif dataEncoder==4 or dataEncoder==6 or dataEncoder==3 then
    --"triple single sequence already flattened representation "
    return data[j]:reshape(self.max_dataseq_l, self.max_prop_namel + self.max_prop_valuel)

  end

end

return edData




