--[[ Return the maxLength, sizes, and non-zero count
  of a baBoxBatch`seq`s ignoring `ignore` words.
--]]
local function getLength(seq, ignore)
  local sizes = torch.IntTensor(seq:size(1)):zero() --in our data seq is a 2D tensor
  local max = 0
  local sum = 0

  for i = 1, seq:size(1) do
    local len = seq[i]:size(1) - seq[i]:eq(ignore):sum(1)[1] --discount the nb of indexes to be ignored
    max = math.max(max, len)
    sum = sum + len
    sizes[i] = len
  end
  return max, sizes, sum
end

--[[ Data management and batch creation.

Batch interface reference [size]:

  * size: number of sentences in the batch [1]
  * sourceLength: max length in source batch [1]
  * sourceSize:  lengths of each source [batch x 1]
  * sourceInput:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * sourceInputFeatures: table of source features sequences
  * sourceInputRev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * sourceInputRevFeatures: table of reversed source features sequences
  * targetLength: max length in source batch [1]
  * targetSize: lengths of each source [batch x 1]
  * targetNonZeros: number of non-ignored words in batch [1]
  * targetInput: input idx's of target (SABCDEPPPPPP) [batch x max]
  * targetInputFeatures: table of target input features sequences
  * targetOutput: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * targetOutputFeatures: table of target output features sequences

 TODO: change name of size => maxlen
--]]

--[[ A batch of property sets to generate.

  Used by the decoder and encoder objects.
--]]
local WikiDataBatch3 = torch.class('WikiDataBatch3')

--[[ Create a batch object.

Parameters:

  -- edBatch: originally created by  ../data2text-align/s2sattn_dta/edData.lua class.
  -- Is a table of tensors containing:
  --     1. Input data property set (src)
  --     2. Target input sequence (tgtIn)
  --     3. Target output sequence (tgtOut)
  --     4. Target alignment labels seaquence (might be empty if not using guided_alignment)
  --     5. Curid and sentence id of the data example. TODO: might review this encoding in case of using the whole abstract!
--]]
function WikiDataBatch3:__init(edBatch)
  edBatch = edBatch[1]
  self.curids = edBatch[5]
  local tgtIn = edBatch[2]
  local tgtOut = edBatch[3]
  local tgtAlnLabels = edBatch[4]

  self.size = tgtIn:size()[1]

  self.sourceLength = edBatch[1]:size(2) --length of input property set (though each property sequence has also a max length)
  self.sourceSize =  torch.IntTensor(edBatch[1]:size(1)):fill(edBatch[1]:size(2))

  self.sourceInput = edBatch[1]

  self.targetInput = tgtIn
  self.targetOutput = tgtOut
  self.targetLabels = tgtAlnLabels

  self.rulTargetLength, self.rulTargetSize, self.rulTargetNonZeros = getLength(tgtIn, 1)
  self.rulTargetLength = tgtIn:size(2) --fixed to this as batch is already padded
                                       --TODO: veryfy how is used by open NMT
  self.targetSize = self.rulTargetSize
  self.targetNonZeros = self.rulTargetNonZeros
  self.targetLength = self.rulTargetLength
  self.targetOffset = 0 -- used for long target stuff

  self.seqLevelBlock = 'block'
end

function WikiDataBatch3:splitIntoPieces(maxBptt)
    self.maxBptt = maxBptt
    self.targetLength = math.min(self.rulTargetLength, maxBptt)
    return math.ceil(self.rulTargetLength/maxBptt)
end

function WikiDataBatch3:nextPiece()
    self.targetOffset = self.targetOffset + self.maxBptt
    self.targetLength = math.min(self.rulTargetLength-self.targetOffset, self.maxBptt)
    self.targetNonZeros = 0 -- so we only count this once...
end

local function addInputFeatures(inputs, featuresSeq, t)
  local features = {}
  for j = 1, #featuresSeq do
    table.insert(features, featuresSeq[j][t])
  end
  if #features > 1 then
    table.insert(inputs, features)
  else
    onmt.utils.Table.append(inputs, features)
  end
end

--[[ Get source batch at timestep `t`. --]]
function WikiDataBatch3:getSourceInput(t)
    assert(false)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.sourceInput[self.inputRow][t]

  if self.batchRowFeats then
      inputs = {inputs, self.batchRowFeats[self.inputRow], self.batchColFeats[t]}
  end

  return inputs
end

-- returns a nRows*srcLen x batchSize tensor
function WikiDataBatch3:getSource()
    return self.sourceInput
end

function WikiDataBatch3:getSourceWords()
    --todo: where/how will be used? see to better adapt.
    --return self.sourceInput:select(2,1):reshape(self.size, self.totalSourceLength)
    return self.sourceInput
end

function WikiDataBatch3:getCellsForExample(b)
    return self.sourceInput
      :sub((b-1)*self.totalSourceLength+1, b*self.totalSourceLength):select(2,1)
end

function WikiDataBatch3:getSourceTriples()
    return self.triples
end

--[[ Get target input batch at timestep `t`. --]]
function WikiDataBatch3:getTargetInput(t)
  local inputs
  if t ~= nil then
    -- If a regular input, return word id, otherwise a table with features.
    inputs = self.targetInput[{{},self.targetOffset + t}]
  else
    inputs = self.targetInput
  end
  return inputs
end

--[[ Get target output batch at timestep `t` (values t+1). --]]
function WikiDataBatch3:getTargetOutput(t)
  local outputs
  if t ~= nil then
    if t == self.seqLevelBlock then
        outputs = self.targetOutput[{{}, {self.targetOffset +1 ,self.targetOffset + self.targetLength }}]
    else
        -- If a regular input, return word id, otherwise a table with features.
        outputs = { self.targetOutput[{{},self.targetOffset + t}] }
    end
  else
    outputs = self.targetOutput
  end
  return outputs
end

--[[ Get target alnLabels batch at timestep `t` (values t+1). --]]
function WikiDataBatch3:getTargetAlnLabels(t)
  local alnLabels
  if t ~= nil then
    if t == self.seqLevelBlock then
        alnLabels = self.targetLabels[{{}, {self.targetOffset +1 ,self.targetOffset + self.targetLength }}]
    else
        -- If a regular input, return word id, otherwise a table with features.
        alnLabels = { self.targetLabels[{{},self.targetOffset + t}] }
    end
  else
    alnLabels = self.targetLabels
  end
  return alnLabels
end

function WikiDataBatch3:getOrigIDs()
    return self.curids
end

function WikiDataBatch3:generalDocumentPos(t)
    return self.targetOffset + t
end

function WikiDataBatch3:blockSampleStart(sampleStart)
    if (sampleStart - self.targetOffset) > 0 then
        return sampleStart - self.targetOffset + 1
    else
        return 1
    end
end

return WikiDataBatch3
