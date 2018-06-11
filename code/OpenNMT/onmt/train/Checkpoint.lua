-- Class for saving and loading models during training.
local Checkpoint = torch.class("Checkpoint")

function Checkpoint:__init(options, model, flatParams, optim, dataset)
  self.options = options
  self.model = model
  self.optim = optim
  self.dataset = dataset
  self.flatParams = flatParams

  self.savePath = self.options.save_model
end

function Checkpoint:save(filePath, info)
  info.learningRate = self.optim:getLearningRate()
  info.optimStates = self.optim:getStates()

  local data = {
    --models = {},
    flatParams = self.flatParams,
    options = self.options,
    info = info,
    dicts = self.dataset.dicts
  }

  --[[for k, v in pairs(self.model) do
     self.model[k] = v:serialize()
  end]]

  torch.save(filePath, data)
end

--[[ Save the model and data in the middle of an epoch sorting the iteration. ]]
function Checkpoint:saveIteration(iteration, epochState, batchOrder, verbose)
  local info = {}
  info.iteration = iteration + 1
  info.epoch = epochState.epoch
  info.epochStatus = epochState:getStatus()
  info.batchOrder = batchOrder

  local filePath = string.format('%s_checkpoint.t7', self.savePath)

  if verbose then
    print('Saving checkpoint to \'' .. filePath .. '\'...')
  end

  -- Succeed serialization before overriding existing file
  self:save(filePath .. '.tmp', info)
  os.rename(filePath .. '.tmp', filePath)
end

function Checkpoint:saveEpoch(validPpl, epochState, verbose, timeStamp)
  local info = {}
  info.validPpl = validPpl
  info.epoch = epochState.epoch + 1
  info.iteration = 1
  info.trainTimeInMinute = epochState:getTime() / 60

  --local filePath = string.format('%s_epoch%d_%.2f.t7', self.savePath, epochState.epoch, validPpl)
  local filePath = string.format('%s_epoch%d_%.2f.t7', self.savePath .. timeStamp, epochState.epoch, validPpl)

  if verbose then
    print('Saving checkpoint to \'' .. filePath .. '\'...')
  end

  self:save(filePath, info)
end

function Checkpoint:deleteEpoch(validPpl, epoch)
  local filePath = string.format('%s_epoch%d_%.2f.t7', self.savePath, epoch, validPpl)
  os.remove(filePath)
end

return Checkpoint
