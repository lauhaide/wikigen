
require 'hdf5'


--[[ Data management and batch creation. Handles data created by `Preprocess.py`. ]]
local WikiDataset2, parent= torch.class("WikiDataset2", "edData")

--[[ Initialize a data object given aligned tables of IntTensors `srcData`
  and `tgtData`.
--]]
function WikiDataset2:__init(opt, data_file, copyGenerate, version, tripV, switch, multilabel)

  parent.__init(self, opt, data_file)

  self.sentHierarch = opt.hdec
  self.tripV = tripV
  self.switch = switch
  self.multilabel = multilabel


  --self.pointers = switch and tgtData.pointers --is this for conditional copy????

  self.copyGenerate = copyGenerate
  self.version = version
end

--[[ Setup up the training data to respect `maxBatchSize`.
-- Create batches up to batch size, our edData does this on cacheMiniBatches() . ]]
function WikiDataset2:setBatchSize(maxBatchSize, dataEncoder, testMode)

  self.batchRange = {}
  self.maxTargetLength = 0
  if testMode then print(" * Return all batches for evaluation") end
  parent.cacheMiniBatches(self, maxBatchSize, dataEncoder, testMode) --todo add method to return created batches ????

end

--[[ Return number of batches. call our edData batch count fc. ]]
function WikiDataset2:batchCount()
  --if self.batchRange == nil then
  --  return 1
  --end
  --return #self.batchRange
  return parent.cachedMBatchesCount(self)
end


--[[ Get `Batch` number `idx`. If nil make a batch of all the data. ]]
function WikiDataset2:getBatch(idx)


  --[[if idx == nil or self.batchRange == nil then
      assert(false)
    return onmt.data.BoxBatch.new(self.srcs, self.srcFeatures, self.tgt,
      self.tgtFeatures, self.maxSourceLength)
  end

  local rangeStart = self.batchRange[idx]["begin"]
  local rangeEnd = self.batchRange[idx]["end"]

  local srcs = {}
  for j = 1, #self.srcs do srcs[j] = {} end
  local tgt = {}
  local triples = {}
  local pointers = {}

  local srcFeatures = {}
  local tgtFeatures = {}

  for i = rangeStart, rangeEnd do
    for j = 1, #self.srcs do
        table.insert(srcs[j], self.srcs[j][i])
    end
    table.insert(tgt, self.tgt[i])

    if self.srcTriples then
        table.insert(triples, self.srcTriples[i]:long())
    end

    if self.switch then
        table.insert(pointers, self.pointers[i])
    end

    if self.srcFeatures[i] then
      table.insert(srcFeatures, self.srcFeatures[i])
    end

    if self.tgtFeatures[i] then
      table.insert(tgtFeatures, self.tgtFeatures[i])
    end
  end
]]


  --need to create a batch of this class? our batches where just tables !!

  local bb
  if self.switch then
      bb = onmt.data.BoxSwitchBatch.new(srcs, srcFeatures, tgt, tgtFeatures,
            self.maxSourceLength, self.colStartIdx, self.nFeatures,
            pointers, self.multilabel)
  elseif self.sentHierarch then
      bb = onmt.data.WikiHierarchicalBatch.new(self.cachedBatches[idx])
  else
    --[[
    bb = onmt.data.BoxBatch3.new(srcs, srcFeatures, tgt, tgtFeatures,
        self.maxSourceLength, self.colStartIdx, self.nFeatures,
        triples, self.tripV)
    ]]
      bb = onmt.data.WikiDataBatch3.new(self.cachedBatches[idx])
  end
  return bb
end

return WikiDataset2
