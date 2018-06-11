--[[ Unit to decode a sequence of output tokens.

     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local SwitchingDecoder, parent = torch.class('onmt.SwitchingDecoder', 'onmt.Sequencer')


--[[ Construct a decoder layer.

Parameters:

  * `inputNetwork` - input nn module.
  * `rnn` - recurrent module, such as [onmt.LSTM](onmt+modules+LSTM).
  * `generator` - optional, an output [onmt.Generator](onmt+modules+Generator).
  * `inputFeed` - bool, enable input feeding.
  * `returnAttnScores` - bool, return unnormalized attn scores at each step
  * `tanhQuery` - bool, add an additional nonlinearity to target state when computing attn
--]]
function SwitchingDecoder:__init(inputNetwork, rnn, generator, inputFeed,
    returnAttnScores, tanhQuery, map, multilabel)
  self.rnn = rnn
  self.inputNet = inputNetwork

  self.args = {}
  self.args.rnnSize = self.rnn.outputSize
  self.args.numEffectiveLayers = self.rnn.numEffectiveLayers

  self.args.inputIndex = {}
  self.args.outputIndex = {}
  self.map = map -- map perplexity computation
  self.multilabel = multilabel

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  self.args.inputFeed = inputFeed
  self.args.returnAttnScores = returnAttnScores
  self.args.tanhQuery = tanhQuery

  parent.__init(self, self:_buildModel())

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generator = generator
  self:add(self.generator)

  self.ptrGenerator = onmt.PointerGenerator(self.args.rnnSize, tanhQuery, false, multilabel)
  self:add(self.ptrGenerator)

  self.switcher = self:_buildSwitcher()
  self:add(self.switcher)

  self:resetPreallocation()
end

--[[ Return a new Decoder using the serialized data `pretrained`. ]]
function SwitchingDecoder.load(pretrained)
  local self = torch.factory('onmt.SwitchingDecoder')()

  self.args = pretrained.args

  parent.__init(self, pretrained.modules[1])
  self.generator = pretrained.modules[2]
  self:add(self.generator)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function SwitchingDecoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

function SwitchingDecoder:resetPreallocation()
  if self.args.inputFeed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end

--[[ Build a default one time-step of the decoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.
--]]
function SwitchingDecoder:_buildModel()
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, self.args.numEffectiveLayers do
    local h0 = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)
  self.args.inputIndex.x = #inputs

  local context = nn.Identity()() -- batchSize x sourceLength x rnnSize
  table.insert(inputs, context)
  self.args.inputIndex.context = #inputs

  local inputFeed
  if self.args.inputFeed then
    inputFeed = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, inputFeed)
    self.args.inputIndex.inputFeed = #inputs
  end

  -- Compute the input network.
  local input = self.inputNet(x)

  -- If set, concatenate previous decoder output.
  if self.args.inputFeed then
    input = nn.JoinTable(2)({input, inputFeed})
  end
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.rnn(states)

  -- The output of a subgraph is a node: split it to access the last RNN output.
  outputs = { outputs:split(self.args.numEffectiveLayers) }

  -- Compute the attention here using h^L as query.
  local attnLayer = onmt.GlobalAttention(self.args.rnnSize, self.args.returnAttnScores, self.args.tanhQuery)
  attnLayer.name = 'decoderAttn'
  local attnOutput, attnScores
  if self.args.returnAttnScores then
    attnOutput, attnScores = attnLayer({outputs[#outputs], context}):split(2)
  else
    attnOutput = attnLayer({outputs[#outputs], context})
  end
  if self.rnn.dropout > 0 then
    attnOutput = nn.Dropout(self.rnn.dropout)(attnOutput)
  end
  table.insert(outputs, attnOutput)
  if self.args.returnAttnScores then
    table.insert(outputs, attnScores)
  end
  return nn.gModule(inputs, outputs)
end

function SwitchingDecoder:_buildSwitcher()
    local switcher = nn.Sequential()
                       :add(nn.ParallelTable()
                              :add(nn.Mean(2))
                              :add(nn.Identity()))
                       :add(nn.JoinTable(2))
                       :add(nn.Linear(2*self.args.rnnSize, self.args.rnnSize))
                       :add(nn.ReLU())
                       --:add(nn.Dropout(0.3))
                       :add(nn.Linear(self.args.rnnSize, 1))
                       :add(nn.Sigmoid())
    return switcher
end


--[[ Mask padding means that the attention-layer is constrained to
  give zero-weight to padding. This is done by storing a reference
  to the softmax attention-layer.

  Parameters:

  * See  [onmt.MaskedSoftmax](onmt+modules+MaskedSoftmax).
--]]
function SwitchingDecoder:maskPadding(sourceSizes, sourceLength)

  local function substituteSoftmax(module)
    if module.name == 'softmaxAttn' then
      local mod
      if sourceSizes ~= nil then
        mod = onmt.MaskedSoftmax(sourceSizes, sourceLength)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmaxAttn'
      mod:type(module._type)
      self.softmaxAttn = mod
      return mod
    else
      return module
    end
  end

  if not self.decoderAttn then
    self.network:apply(function (layer)
      if layer.name == 'decoderAttn' then
        self.decoderAttn = layer
      end
    end)
  end
  self.decoderAttn:replace(substituteSoftmax)

  if not self.decoderAttnClones then
    self.decoderAttnClones = {}
  end
  for t = 1, #self.networkClones do
    if not self.decoderAttnClones[t] then
      self:net(t):apply(function (layer)
        if layer.name == 'decoderAttn' then
          self.decoderAttnClones[t] = layer
        end
      end)
    end
    self.decoderAttnClones[t]:replace(substituteSoftmax)
  end
end

function SwitchingDecoder:remember()
    self._remember = true
end

function SwitchingDecoder:forget()
    self._remember = false
end

function SwitchingDecoder:resetLastStates()
    self.lastStates = nil
end

--[[ Run one step of the decoder.

Parameters:

  * `input` - input to be passed to inputNetwork.
  * `prevStates` - stack of hidden states (batch x layers*model x rnnSize)
  * `context` - encoder output (batch x n x rnnSize)
  * `prevOut` - previous distribution (batch x #words)
  * `t` - current timestep

Returns:

 1. `out` - Top-layer hidden state.
 2. `states` - All states.
--]]
function SwitchingDecoder:forwardOne(input, prevStates, context, prevOut, t)
  local inputs = {}

  -- Create RNN input (see sequencer.lua `buildNetwork('dec')`).
  onmt.utils.Table.append(inputs, prevStates)
  table.insert(inputs, input)
  table.insert(inputs, context)
  local inputSize
  if torch.type(input) == 'table' then
    inputSize = input[1]:size(1)
  else
    inputSize = input:size(1)
  end

  if self.args.inputFeed then
    if prevOut == nil then
      table.insert(inputs, onmt.utils.Tensor.reuseTensor(self.inputFeedProto,
                                                         { inputSize, self.args.rnnSize }))
    elseif self.args.returnAttnScores then -- prevOut is a table
      table.insert(inputs, prevOut[1])
    else
      table.insert(inputs, prevOut)
    end
  end

  -- Remember inputs for the backward pass.
  if self.train then
    self.inputs[t] = inputs
  end

  local outputs = self:net(t):forward(inputs) -- [states; attnOutput; (attnScores)]
  local attnScoresLoc = self.args.returnAttnScores and 1 or 0
  local out = outputs[#outputs - attnScoresLoc]
  local states = {}
  for i = 1, #outputs - 1 - attnScoresLoc do
    table.insert(states, outputs[i])
  end
  if self.args.returnAttnScores then
    out = {out, outputs[#outputs]}
  end

  return out, states
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - `Batch` object
  * `encoderStates` -
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function SwitchingDecoder:forwardAndApply(batch, encoderStates, context, func)
  -- TODO: Make this a private method.

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         { batch.size, self.args.rnnSize })
  end

  local states, prevOut
  if self._remember and self.lastStates then
      prevOut = self.lastStates[#self.lastStates]
      states = {} -- could probably really just pop
      for i = 1, #self.lastStates-1 do
          table.insert(states, self.lastStates[i])
      end
  else
      states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)
  end

  --local states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)

  --local prevOut

  for t = 1, batch.targetLength do
    prevOut, states = self:forwardOne(batch:getTargetInput(t), states, context, prevOut, t)
    func(prevOut, t)
  end

  if self._remember then
      self.lastStates = self:net(batch.targetLength).output
  end
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - a `Batch` object.
  * `encoderStates` - a batch of initial decoder states (optional) [0]
  * `context` - the context to apply attention to.

  Returns: Table of top hidden state for each timestep.
--]]
function SwitchingDecoder:forward(batch, encoderStates, context)
  encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })
  if self.train then
    self.inputs = {}
  end

  local outputs = {}

  self:forwardAndApply(batch, encoderStates, context, function (out)
    table.insert(outputs, out)
  end)

  return outputs
end

--[[ Compute the backward update.

Parameters:

  * `batch` - a `Batch` object
  * `outputs` - expected outputs
  * `criterion` - a single target criterion object

  Note: This code runs both the standard backward and criterion forward/backward.
  It returns both the gradInputs and the loss.
  -- ]]
function SwitchingDecoder:backward(batch, outputs, criterion, ctxLen, dummy, switchCrit, ptrCrit)
  assert(not self.args.returnAttnScores) -- otherwise have to get grads from generated shit and zero out differently
  local decLayers = self.args.numEffectiveLayers + 1
  if self.args.returnAttnScores then
    decLayers = decLayers + 1
  end

  local attnScoresLoc = self.args.returnAttnScores and 1 or 0

  local layerSizes = {}
  if self.args.returnAttnScores then -- last layer has different length than all the others
      for i = 1, decLayers-1 do
          table.insert(layerSizes, {batch.size, self.args.rnnSize})
      end
      table.insert(layerSizes, {batch.size, batch.totalSourceLength})
  else
      layerSizes = {batch.size, self.args.rnnSize} -- will be used for every layer
  end

  if self.gradOutputsProto == nil then
    self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(decLayers,
                                                              self.gradOutputProto,
                                                              layerSizes)
  end

  local ctxLen = ctxLen or batch.sourceLength -- for back compat
  local gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                             layerSizes)
  local gradContextInput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                                         { batch.size, ctxLen, self.args.rnnSize })

  local loss, switchLoss, ptrLoss = 0, 0, 0

  local context = self.inputs[1][self.args.inputIndex.context]

  local ptrCrit = ptrCrit or criterion

  for t = batch.targetLength, 1, -1 do
    local finalLayer = self:net(t).output[self.args.numEffectiveLayers]

    local zs = batch:getZs(t)
    local zpreds = self.switcher:forward({context, finalLayer})
    switchLoss = switchLoss + switchCrit:forward(zpreds, zs)
    local zpredGradOut = switchCrit:backward(zpreds, zs)
    zpredGradOut:div(batch.totalSize)
    local decSwitchGO = self.switcher:backward({context, finalLayer}, zpredGradOut)

    local ptrPreds = self.ptrGenerator:forward({context, finalLayer})
    ptrLoss = ptrLoss + ptrCrit:forward(ptrPreds, batch:getPointerTargets(t))
    local ptrPredGradOut = ptrCrit:backward(ptrPreds, batch:getPointerTargets(t))
    ptrPredGradOut:div(batch.totalSize)
    for b = 1, batch.size do
        if zs[b] ~= 1 then -- not a copy
            ptrPredGradOut[b]:zero()
        end
    end
    local decPtrGO = self.ptrGenerator:backward({context, finalLayer}, ptrPredGradOut)

    gradContextInput:add(decSwitchGO[1])
    gradContextInput:add(decPtrGO[1])

    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    local pred = self.generator:forward(outputs[t])
    local output = batch:getTargetOutput(t)
    -- training loss we return will be wrong b/c disregards zs, but who cares
    loss = loss + criterion:forward(pred, output)

    -- Compute the criterion gradient.
    local genGradOut = criterion:backward(pred, output)
    for j = 1, #genGradOut do
      genGradOut[j]:div(batch.totalSize)
      for b = 1, batch.size do
          if zs[b] == 1 then -- a copy
              genGradOut[j][b]:zero()
          end
      end
    end

    -- Compute the final layer gradient.
    local decGradOut = self.generator:backward(outputs[t], genGradOut)

    local outputLayers = 1
    if torch.type(decGradOut) == 'table' then
      outputLayers = #decGradOut
    else
      decGradOut = {decGradOut}
    end

    for j = 1, outputLayers do
        gradStatesInput[decLayers-outputLayers+j]:add(decGradOut[j])
    end

    gradStatesInput[decLayers-outputLayers]:add(decSwitchGO[2])
    gradStatesInput[decLayers-outputLayers]:add(decPtrGO[2])


    -- Compute the standard backward.
    local gradInput = self:net(t):backward(self.inputs[t], gradStatesInput)

    -- Accumulate encoder output gradients.
    gradContextInput:add(gradInput[self.args.inputIndex.context])

    for j = 1, outputLayers do
      gradStatesInput[decLayers-outputLayers+j]:zero()
    end

    -- Accumulate previous output gradients with input feeding gradients.
    if self.args.inputFeed and t > 1 then
      gradStatesInput[decLayers - attnScoresLoc]:add(gradInput[self.args.inputIndex.inputFeed])
    end

    -- Prepare next decoder output gradients.
    for i = 1, #self.statesProto do
      gradStatesInput[i]:copy(gradInput[i])
    end
  end

  if batch.targetOffset > 0 then -- hack so that we only backprop thru final encoder state at beginning
      for i = 1, #self.statesProto do
          gradStatesInput[i]:zero()
      end
  end

  return gradStatesInput, gradContextInput, loss, switchLoss, ptrLoss
end

--[[ Compute the loss on a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.
  * `criterion` - a pointwise criterion.

--]]
function SwitchingDecoder:computeLoss(batch, encoderStates, context, criterion)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local loss = 0
  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    local finalLayer = self:net(t).output[self.args.numEffectiveLayers]
    local zpreds = self.switcher:forward({context, finalLayer})
    local ptrPreds = self.ptrGenerator:forward({context, finalLayer})
    local pred = self.generator:forward(out)[1]
    local output = batch:getTargetOutput(t)[1]
    for b = 1, batch.size do
        if output[b] ~= onmt.Constants.PAD then
            if self.map then -- just take argmax prob
                if zpreds[b][1] >= 0.5 then -- a copy
                    pred[b]:zero()
                    --  marginalize over all copies of same word
                    if not self.multilabel then
                        ptrPreds[b]:exp()
                    end
                    pred[b]:indexAdd(1, batch:getCellsForExample(b), ptrPreds[b])
                    pred[b]:log()
                    loss = loss - pred[b][output[b]]
                else
                    loss = loss - pred[b][output[b]]
                end
            else -- truly marginalize
                pred[b]:add(math.log(1-zpreds[b][1]))
                if self.multilabel then
                    ptrPreds[b]:mul(zpreds[b][1])
                else
                    ptrPreds[b]:add(math.log(zpreds[b][1]))
                end
                pred[b]:exp()
                if not self.multilabel then
                    ptrPreds[b]:exp()
                end
                pred[b]:indexAdd(1, batch:getCellsForExample(b), ptrPreds[b])
                pred[b]:log()
                loss = loss - pred[b][output[b]]
            end
        end
    end

    --loss = loss + criterion:forward(pred, output)
  end)

  return loss
end


function SwitchingDecoder:computeActualLoss(batch, encoderStates, context, criterion, switchCrit, ptrCrit)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local loss = 0
  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    local finalLayer = self:net(t).output[self.args.numEffectiveLayers]

    local zs = batch:getZs(t)
    local zpreds = self.switcher:forward({context, finalLayer})
    local switchLoss = switchCrit:forward(zpreds, zs)
    loss = loss + switchLoss

    local ptrPreds = self.ptrGenerator:forward({context, finalLayer})
    local ptrTargs = batch:getPointerTargets(t)
    --local ptrLoss = ptrCrit:forward(ptrPreds, batch:getPointerTargets(t))

    local pred = self.generator:forward(out)[1]
    local output = batch:getTargetOutput(t)[1]
    for b = 1, batch.size do
        if output[b] ~= onmt.Constants.PAD then
            if zs[b] == 1 then -- just take argmax prob
                if self.multilabel then
                    assert(false) -- don't wanna do it
                else
                    loss = loss - ptrPreds[b][ptrTargs[b]]
                end
            else
                loss = loss - pred[b][output[b]]
            end
        end
    end

    --loss = loss + criterion:forward(pred, output)
  end)

  return loss
end

--[[ Compute the score of a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.

--]]
function SwitchingDecoder:computeScore(batch, encoderStates, context)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local score = {}

  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    local pred = self.generator:forward(out)
    for b = 1, batch.size do
      if t <= batch.targetSize[b] then
        score[b] = (score[b] or 0) + pred[1][b][batch.targetOutput[t][b]]
      end
    end
  end)

  return score
end

function SwitchingDecoder:greedyFixedFwd(batch, encoderStates, context, probBuf)
    if not self.greedy_inp then
        self.greedy_inp = torch.CudaTensor()
        self.maxes = torch.CudaTensor()
        self.argmaxes = torch.CudaLongTensor()
    end
    local PAD, EOS = onmt.Constants.PAD, onmt.Constants.EOS
    self.greedy_inp:resize(batch.targetLength+1, batch.size):fill(PAD)
    self.maxes:resize(batch.size, 1)
    self.argmaxes:resize(batch.size, 1)

    if self.statesProto == nil then
      self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                           self.stateProto,
                                                           { batch.size, self.args.rnnSize })
    end

    local states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)

    local prevOut

    self.greedy_inp[1]:copy(batch:getTargetInput(1)) -- should be start token
    for t = 1, batch.targetLength do
      prevOut, states = self:forwardOne(self.greedy_inp[t], states, context, prevOut, t)
      local preds = self.generator:forward(prevOut)
      torch.max(self.maxes, self.argmaxes, preds[1], 2)
      if probBuf then
          probBuf[t]:copy(self.maxes:view(-1))
      end
      self.greedy_inp[t+1]:copy(self.argmaxes:view(-1))
    end
    return self.greedy_inp
end
