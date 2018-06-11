--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class("EpochState")

--[[ Initialize for epoch `epoch` and training `status` (current loss)]]
function EpochState:__init(epoch, numIterations, learningRate, lastValidPpl, status)
  self.epoch = epoch
  self.numIterations = numIterations
  self.learningRate = learningRate
  self.lastValidPpl = lastValidPpl

  if status ~= nil then
    self.status = status
  else
    self.status = {}
    self.status.trainNonzeros = 0
    self.status.trainLoss = 0
    self.status.recLoss = 0
  end

  self.timer = torch.Timer()
  self.numWordsSource = 0
  self.numWordsTarget = 0

  self.minFreeMemory = 100000000000
end

--[[ Update training status. Takes `batch` (described in data.lua) and last loss.]]
function EpochState:update(batch, loss, recloss)
  self.numWordsSource = self.numWordsSource + batch.size * batch.sourceLength
  self.numWordsTarget = self.numWordsTarget + batch.size * batch.targetLength
  self.status.trainLoss = self.status.trainLoss + loss
  if recloss then
      self.status.recLoss = self.status.recLoss + recloss
  end
  self.status.trainNonzeros = self.status.trainNonzeros + batch.targetNonZeros
end


--[[ Update training status. Takes `batch` (described in data.lua) and last loss.]]
function EpochState:updateRL(batch, loss, nonRLNonZeros)
  self.numWordsSource = self.numWordsSource + batch.size * batch.sourceLength
  self.numWordsTarget = self.numWordsTarget + batch.size * batch.targetLength
  self.status.trainLoss = self.status.trainLoss + loss
  self.status.trainNonzeros = self.status.trainNonzeros + nonRLNonZeros
end

--[[ Log to status stdout. ]]
function EpochState:log(batchIndex, json, fd)
  if json then
    local freeMemory = onmt.utils.Cuda.freeMemory()
    if freeMemory < self.minFreeMemory then
      self.minFreeMemory = freeMemory
    end

    local obj = {
      time = os.time(),
      epoch = self.epoch,
      iteration = batchIndex,
      totalIterations = self.numIterations,
      learningRate = self.learningRate,
      trainingPerplexity = self:getTrainPpl(),
      freeMemory = freeMemory,
      lastValidationPerplexity = self.lastValidPpl,
      processedTokens = {
        source = self.numWordsSource,
        target = self.numWordsTarget
      }
    }

    onmt.utils.Log.logJson(obj)
  else
    local timeTaken = self:getTime()

    local stats = ''
    stats = stats .. string.format('Epoch %d ; ', self.epoch)
    stats = stats .. string.format('Iter %d/%d ; ', batchIndex, self.numIterations)
    stats = stats .. string.format('LR %.4f ; ', self.learningRate)
    stats = stats .. string.format('Target tokens/s %d ; ', self.numWordsTarget / timeTaken)
    stats = stats .. string.format('Loss %.2f ; ', self.status.trainLoss)
    stats = stats .. string.format('PPL %.2f ; ', self:getTrainPpl())
    if self.status.recLoss ~= 0 then
        stats = stats .. string.format('RLoss %.3f', self.status.recLoss/self.status.trainNonzeros)
    end
    print(stats)
    if fd then
      fd:write(stats)
      fd:flush()
    end

  end
end

function EpochState:getTrainPpl()
  return math.exp(self.status.trainLoss / self.status.trainNonzeros)
end

function EpochState:getTime()
  return self.timer:time().real
end

function EpochState:getStatus()
  return self.status
end

function EpochState:getMinFreememory()
  return self.minFreeMemory
end

return EpochState
