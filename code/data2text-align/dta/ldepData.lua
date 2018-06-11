--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 17/05/17
-- Time: 14:52
-- To change this template use File | Settings | File Templates.
--



local ldepData = torch.class("ldepData")

function ldepData:__init(opt, data_file)

  print("reading file " .. data_file)
  local f = hdf5.open(data_file, 'r')

  self.local_sentences = f:read('sentence'):all()
  self.local_labels = f:read('label'):all()

  self.cachedBatches = {}
  self.nbCachedMiniBatches = 0

  self:cacheMultitaskBatches()

  print(tostring(self.nbCachedMiniBatches) .. " cached mini-batches...")
end

function ldepData:tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end

--[[informs the nb of current exiting minibatches]]
function ldepData:cachedMBatchesCount()
  return self.nbCachedMiniBatches
end

--[[total nb of buckets in which elements are stored]]
function ldepData:getBucketCount()
  if not (self:tablelength(self.local_sentences) == self:tablelength(self.local_labels)) then
    print("WARNING: data inconsistency, different nb of text and data buckets!")
  end
  return self:tablelength(self.local_labels)
end

function ldepData:getBucket(i)
  local sentence = {}
  local targetLabel = {}
  for k, v in pairs(self.local_sentences) do
    if tonumber(k)==i then
      table.insert(sentence, localize(v))
    end
  end
  for k, v in pairs(self.local_labels) do
    if tonumber(k)==i then
      table.insert(targetLabel, localize(v))
    end
  end
  return {sentence, targetLabel}
end

function ldepData:cacheMultitaskBatches()
  print("generating multitask training minibatches...")

  self.nbCachedMiniBatches = self:getBucketCount()
  local bucket_b, sents, labels
  for b=1,self:getBucketCount() do
    sents, labels = unpack(self:getBucket(b))
    table.insert(self.cachedBatches, {sents[1], labels[1]})
  end
end

function ldepData:getMultitasMiniBatches()
  return self.cachedBatches
end
