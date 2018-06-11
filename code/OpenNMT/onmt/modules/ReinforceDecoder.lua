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


require 'LMScorer'
require 'WAScorerB1'
require 'WAScorerCombined'
require 'WAScorerCombinedFpLrG'
require 'WAScorerCombined2'
require 'WAScorerCombinedFpLrGSelective'


local ReinforceDecoder, parent = torch.class('onmt.ReinforceDecoder', 'onmt.Sequencer')


--[[ Construct a decoder layer.

Parameters:

  * `inputNetwork` - input nn module.
  * `rnn` - recurrent module, such as [onmt.LSTM](onmt+modules+LSTM).
  * `generator` - optional, an output [onmt.Generator](onmt+modules+Generator).
  * `inputFeed` - bool, enable input feeding.
  * `returnAttnScores` - bool, return unnormalized attn scores at each step
  * `tanhQuery` - bool, add an additional nonlinearity to target state when computing attn
  * `maxTargetLen` - number, max length of target sequences
  * `maxBatchSize` -  number, max batch size
  * `minimumSamplingPos` - number, minimum position up to where sample
  * `blockAlns` - bool, whether the reward uses the alignments per block or whole document
--]]
function ReinforceDecoder:__init(inputNetwork, rnn, generator, inputFeed,
    returnAttnScores, tanhQuery,  maxTargetLen, maxBatchSize, minimumSamplingPos, blockAlns)
  self.rnn = rnn
  self.inputNet = inputNetwork

  self.args = {}
  self.args.rnnSize = self.rnn.outputSize
  self.args.numEffectiveLayers = self.rnn.numEffectiveLayers
  self.args.maxLenProp = 1
  self.args.seqLen  = maxTargetLen * self.args.maxLenProp
  self.args.batchSize = maxBatchSize
  self.args.minimumSamplingPos = minimumSamplingPos
  self.args.blockAlns = blockAlns

  self.args.inputIndex = {}
  self.args.outputIndex = {}

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  self.args.inputFeed = inputFeed
  self.args.returnAttnScores = returnAttnScores
  self.args.tanhQuery = tanhQuery

  parent.__init(self, self:_buildModel())

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generatorClones = {}
  self.generatorClones[1] = generator
  self:add(self.generatorClones[1])
    for i=2 , self.args.seqLen do
     self.generatorClones[i] = generator:clone('weight', 'gradWeight', 'bias', 'gradBias')
    end


  self.args.novocab = self.generatorClones[1].outputSize

  -- expected rewards generator
  self:_createRewardPredictors()
  self.sampleStart = math.huge --to use what's given as parameter or never start
  self.rf_criterion = localize( nn.ReinforceCriterion(self.sampleStart, self.args.seqLen, self.args.batchSize) )
  self.postParametersInitialization = function()
      for i = 1, self.args.seqLen do
        print("setting reward predictors bias and weights")
        self.cum_reward_predictors[i]:postParametersInitialization()
    end
  end

  self:resetPreallocation()
end

function ReinforceDecoder:getParent()
  return parent
end

--[[ Return a new Decoder using the serialized data `pretrained`. ]]
function ReinforceDecoder.load(pretrained)
    --TODO rewrite this piece of code, now in this class generators are cloned
  print("Unimplemented methdod. / Out of date code.")
  os.exit()
end

--[[ Return data to serialize. ]]
function ReinforceDecoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

function ReinforceDecoder:resetPreallocation()
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


function ReinforceDecoder:_createRewardPredictors()
  --reward predictors are not clonned !
  self.cum_reward_predictors = {}
  for i = 1, self.args.seqLen do
    local ln  = nn.Linear(self.args.rnnSize, 1)
    ln.postParametersInitialization = function()
      --to allow needed reward generator particular initialisation)
      ln.bias:fill(0.01)
      ln.weight:fill(0)
      end
    self.cum_reward_predictors[i] = localize(ln)
  end

  --to be seen if we can use this prototypes
  self.reward = localize( torch.zeros(self.args.seqLen, self.args.batchSize) )
  self.reward_mask = localize( torch.zeros(self.args.seqLen, self.args.batchSize) )
  self.df_preds = localize( torch.Tensor(self.args.batchSize, self.args.novocab) )
  self.dummy_df_y_rf = localize( torch.zeros(self.args.batchSize, self.args.seqLen) )

end


function ReinforceDecoder:copyFrom(tmpModelDecoder)

  local p, _ = self.network:getParameters()
  local tmpP, _ = tmpModelDecoder.network:getParameters()
  p:copy(tmpP)

  if torch.type(tmpModelDecoder.generator) == 'onmt.Generator' then
    print(" * copying generator weights from " .. torch.type(tmpModelDecoder.generator))
    for i=1 , self.args.seqLen do
      self.generatorClones[i].net.forwardnodes[3].data.module.weight:copy(tmpModelDecoder.generator.net.modules[1].weight)
      self.generatorClones[i].net.forwardnodes[3].data.module.bias:copy(tmpModelDecoder.generator.net.modules[1].bias)
    end
  elseif torch.type(tmpModelDecoder.generator) == 'onmt.GuidedAlnGenerator' then
    print(" * copying generator weights from " .. torch.type(tmpModelDecoder.generator))
    --for i=1 , self.args.seqLen do
      self.generatorClones[1].net.forwardnodes[3].data.module.weight:copy(tmpModelDecoder.generator.net.modules[1].modules[1].weight)
      self.generatorClones[1].net.forwardnodes[3].data.module.bias:copy(tmpModelDecoder.generator.net.modules[1].modules[1].bias)
    --end
  end
end


function ReinforceDecoder:setSampleStart(start)
  --update only if the start sampling position is not lower than a minumum sequence position
  if start >= self.args.minimumSamplingPos then
    self.sampleStart = start
  else
    self.sampleStart = self.args.minimumSamplingPos
  end
end


function ReinforceDecoder:getSampleStart()
  return self.sampleStart
end


function ReinforceDecoder:loadLMScorer(lm_path, lmWeight)
  self.lm_scorer = LMScorer(lm_path, self.args.batchSize)
  self.args.lmWeight= lmWeight
end

function ReinforceDecoder:loadWAScorer(modelPath, tgtdict, waWeight, warType)
self.args.lmWeight = 1
  if warType == 1 then
    print(" * Bleu1 reward")
    self.wa_scorer = WAScorerB1(tgtdict, self.args.batchSize)
  elseif warType == 2 then
    print(" * F-measure with (Bleu1, recall) ")
    self.wa_scorer = WAScorerCombined(tgtdict, self.args.batchSize)
  elseif warType == 7 then
    print(" * r1= LOCAL Bleu1 and r2= GLOBAL recall ")
    self.wa_scorer = WAScorerCombined2(tgtdict, self.args.batchSize)
  elseif warType == 8 then
    print(" * F-measure with (LOCAL Bleu1, GLOBAL recall) ")
    self.wa_scorer = WAScorerCombinedFpLrG(tgtdict, self.args.batchSize)
      elseif warType == 9 then
    print(" * F-measure with (LOCAL Bleu1, GLOBAL recall) with selective/constrained RL.")
    self.wa_scorer = WAScorerCombinedFpLrGSelective(tgtdict, self.args.batchSize)
  else
    print("Alignment reward function underspecified.")
    os.exit()
  end
  self.args.waWeight = waWeight
  self.args.warType = warType
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
function ReinforceDecoder:_buildModel()
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

--[[ Mask padding means that the attention-layer is constrained to
  give zero-weight to padding. This is done by storing a reference
  to the softmax attention-layer.

  Parameters:

  * See  [onmt.MaskedSoftmax](onmt+modules+MaskedSoftmax).
--]]
function ReinforceDecoder:maskPadding(sourceSizes, sourceLength)

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

function ReinforceDecoder:remember()
    self._remember = true
end

function ReinforceDecoder:forget()
    self._remember = false
end

function ReinforceDecoder:resetLastStates()
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
function ReinforceDecoder:forwardOne(input, prevStates, context, prevOut, t)
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
    out = {out, outputs[#outputs] }
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

function ReinforceDecoder:forwardAndApply(batch, encoderStates, context, func)
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

  for t = 1, batch.targetLength do
    prevOut, states = self:forwardOne(batch:getTargetInput(t), states, context, prevOut, t)
    func(prevOut, t)
  end

  if self._remember then
      self.lastStates = self:net(batch.targetLength).output
  end
end


--[[Compute all forward steps with sampling for RL evaluation
Relies on forwardOne()
  Parameters:

  * `batch` - `Batch` object
  * `encoderStates` -
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function ReinforceDecoder:forwardSample(batch, encoderStates, context, greedy)
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

  local out_sample, eRewa, pred

  --get real target length or RL preferred sample length
  local Ty = batch.targetLength
  --local Ty_rf = math.min( torch.round(batch.targetLength * 0.8), batch.targetLength )
  --local Ty_rf = batch.targetLength
  local Ty_rf = math.min( torch.round(batch.targetLength * self.args.maxLenProp), batch.targetLength )
  local maxTy = batch:blockSampleStart(self.sampleStart) > Ty and Ty or Ty_rf
  local y_rf = localize( torch.ones(batch.size, maxTy + 1) )
  y_rf[{ {}, 1 }] = batch:getTargetInput(1)

  for t = 1, maxTy do

    prevOut, states = self:forwardOne(y_rf[{ {}, t }], states, context, prevOut, t)
    pred, out_sample = unpack(self.generatorClones[1]:forward(prevOut))

    if not greedy then
      y_rf[{ {}, t+1 }] = out_sample:squeeze()
    else
      local maxv, maxi = pred:max(2)
      y_rf[{ {}, t+1 }] = maxi:squeeze():type('torch.CudaTensor')
    end

  end

  if self._remember then
      self.lastStates = self:net(maxTy).output
  end

  return y_rf
end

function ReinforceDecoder:forwardAndApplyRL(batch, encoderStates, context, func)
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

  local out_sample, eRewa, pred

  --get real target length or RL preferred sample length
  local Ty = batch.targetLength
  local Ty_rf = math.min( torch.round(batch.targetLength * self.args.maxLenProp), batch.targetLength )
  local maxTy, redSeq
  if ( batch:blockSampleStart(self.sampleStart) > Ty or batch:blockSampleStart(self.sampleStart) > Ty_rf ) then
    maxTy =  Ty
    redSeq = false --reduced sequence
  else
    maxTy =  Ty_rf
    redSeq = true
  end
  --y_rf will contain both y_rf_IN and y_rf_OUT sequences with BOS and EOS
  local y_rf = localize( torch.ones(batch.size, maxTy + 1) )
  y_rf[{ {}, 1 }] = batch:getTargetInput(1)

  for t = 1, maxTy do
    prevOut, states = self:forwardOne(y_rf[{ {}, t }], states, context, prevOut, t)
    eRewa = self.cum_reward_predictors[t]:forward(prevOut)
    pred, out_sample = unpack(self.generatorClones[t]:forward(prevOut))

    --decide whether to start using sampled next inputs
    --should consider block offset for multi-sentence texts
    --as sampleStart is defined at the document as entired sequence
    if t < batch:blockSampleStart(self.sampleStart) or not redSeq then
      y_rf[{ {}, t+1 }] = batch:getTargetOutput(t)[1] --this returns the [batch x 1] tensor within a table as neede by criterions
    else
      y_rf[{ {}, t+1 }] = out_sample:squeeze()
    end

    func(prevOut, pred, t, eRewa)
  end

  if self._remember then
      self.lastStates = self:net(maxTy).output
  end

  return y_rf
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - a `Batch` object.
  * `encoderStates` - a batch of initial decoder states (optional) [0]
  * `context` - the context to apply attention to.

  Returns: Table of top hidden state for each timestep.
--]]
function ReinforceDecoder:forward(batch, encoderStates, context)
  encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })
  if self.train then
    self.inputs = {}
  end

  local outputs = {} --decoder outputs
  local yPreds = {} --generator predictions
  local expected_rewards = {} -- compute expected rewards

  --before doing a forward pass of the batch, set the RL criterion start sample variable
  self.rf_criterion:setSampleStart(batch:blockSampleStart(self.sampleStart))

  --y_rf contains the sampled target output sequences
  local y_rf = self:forwardAndApplyRL(batch, encoderStates, context, function (out, ypred, t, rew)
    outputs[t] = out
    yPreds[t] = ypred
    expected_rewards[t] = rew
  end)

  return {outputs, yPreds, expected_rewards, y_rf}
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

--[[ Compute the backward update.

Parameters:

  * `batch` - a `Batch` object
  * `outputs` - expected outputs
  * `criterion` - a single target criterion object

  Note: This code runs both the standard backward and criterion forward/backward.
  It returns both the gradInputs and the loss.
  -- ]]
function ReinforceDecoder:backward(batch, allOutputs, criterion, ctxLen)

  local outputs, yPreds, expected_rewards, y_rf = unpack(allOutputs) --recover decoder outputs and generator predictions

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

  -- ** All RL stuff here ** --
      -- * see why takes -1 and see how not to have this code repeated here and in forwardAndApply* --
  local Ty = batch.targetLength
  local Ty_rf = math.min( torch.round(batch.targetLength * self.args.maxLenProp), batch.targetLength )
  local maxTy, redSeq
  if ( batch:blockSampleStart(self.sampleStart) > Ty or batch:blockSampleStart(self.sampleStart) > Ty_rf ) then
    maxTy =  Ty
    redSeq = false
  else
    maxTy =  Ty_rf
    redSeq = true
  end

  -- ** --


  -- RL reward stuff, which needs a complete forward pass including predictions and actual output y sequence
  local rf_loss, n_seq, df_y_rf, df_expected_rewards
  local end_pos = maxTy

  if batch:blockSampleStart(self.sampleStart) <= maxTy and batch:blockSampleStart(self.sampleStart) <= Ty_rf then

    --at which possition does nll training ends
    end_pos = batch:blockSampleStart(self.sampleStart) - 1

    self.reward:resize(y_rf:size()):zero()
    self.reward_mask:resize(y_rf:size()):zero()
    -- reinforce criterion in Action

    -- * Content coverage reward
    local ori_sents
    if self.args.warType == 8 then
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(batch.seqLevelBlock),
                                    batch:getTargetAlnLabels(batch.seqLevelBlock),
                                    batch:getTargetOutput(), batch:getTargetAlnLabels(),
                                    y_rf, self.reward, self.reward_mask, batch:blockSampleStart(self.sampleStart))
    else
      if self.args.blockAlns then
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(batch.seqLevelBlock),
                                    batch:getTargetAlnLabels(batch.seqLevelBlock), y_rf, self.reward, self.reward_mask)
      else --uses the whole document as target alns
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(),
                                    batch:getTargetAlnLabels(), y_rf, self.reward, self.reward_mask)
      end
    end

    -- combine all rewards together into self.reward
    for i = 1, self.reward:size(1) do
      for j = self.reward:size(2), 1, -1 do
        if self.reward[{ i, j }] ~= 0 then
          self.reward[{ i, j }] = self.reward[{ i, j }] * self.args.waWeight
          break
        end
      end
    end

    inspect(self.reward:sum(), "self.reward")

    rf_loss, n_seq = self.rf_criterion:forward({y_rf, expected_rewards, self.reward, self.reward_mask})
    df_y_rf, df_expected_rewards, expr, cumr = unpack( self.rf_criterion:backward({y_rf, expected_rewards, self.reward, self.reward_mask}) )
    local ok = inspect(df_y_rf:sum(), "df_y_rf")
    if not ok then
      --print indices of nans
      for ii=1, #expr do
        print("t= ", ii, expr[ii]:ne(expr[ii]):max(2):sum())
      end
      print(cumr:sum()) --this has a nan value
      os.exit()
    end


  else
    rf_loss, n_seq = 0, 0
    self.dummy_df_y_rf:resize(y_rf:size()):zero()
    df_y_rf = self.dummy_df_y_rf
  end

  local loss = 0

  for t = maxTy, 1, -1 do
    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    --local pred = self.generator:forward(outputs[t])

    local genGradOut
    if t < batch:blockSampleStart(self.sampleStart) or not redSeq then
      -- * still need to do criterion fw/bw * --

      local tgtOutput = batch:getTargetOutput(t)
      local oli = criterion:forward({yPreds[t]}, tgtOutput)
      inspect(yPreds[t]:sum(), "preds")
      loss = loss + oli
      -- Compute the criterion gradient.
      genGradOut = criterion:backward({yPreds[t]}, tgtOutput)
    else
      -- * higher level rf criterion already evaluated, need to bw on reward predictors * --
      local dec_hs_hat_t = self.cum_reward_predictors[t]:backward(outputs[t], df_expected_rewards[t])
      genGradOut = {self.df_preds:resizeAs(yPreds[t]):zero()}
    end

    for j = 1, #genGradOut do
      genGradOut[j]:div(batch.totalSize)
    end

    -- Compute the final layer gradient.
    -- Take from the y_rf gradient the part that corresponds to the targetOutput sequence
    local decGradOut = self.generatorClones[t]:backward(outputs[t], {genGradOut[1], df_y_rf[{{}, {2, maxTy}}]}) --df_y_rf is already batch normalised

    local outputLayers = 1
    if torch.type(decGradOut) == 'table' then
      outputLayers = #decGradOut
    else
      decGradOut = {decGradOut}
    end

    for j = 1, outputLayers do
        gradStatesInput[decLayers-outputLayers+j]:add(decGradOut[j])
    end

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

  local nllTokens = end_pos > 0 and y_rf[{{}, {2, end_pos+1}}]:ne(1):sum() or 0
  local lossInfo = {loss, rf_loss, n_seq, nllTokens}
  return gradStatesInput, gradContextInput, lossInfo
end

--[[ Compute the loss on a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.
  * `criterion` - a pointwise criterion.

--]]
function ReinforceDecoder:computeLoss(batch, encoderStates, context, criterion)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local loss = 0
  self:forwardAndApply(batch, encoderStates, context, function (out, t)

    --local pred = unpack(self.generator:forward(out))
    local pred = unpack(self.generatorClones[1]:forward(out))
    local output = batch:getTargetOutput(t)
    loss = loss + criterion:forward({pred}, output)
  end)

  return loss
end


--[[ Compute Reinforce losses on a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.
  * `criterion` - a pointwise criterion.

--]]
function ReinforceDecoder:computeSampleLoss(batch, encoderStates, context, greedy)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

--[[  local outputs = {} --decoder outputs
  local yPreds = {} --generator predictions
  --local yRefs = {} --actual references
  local expected_rewards = {} -- compute expected rewards ]]
  local reward_coverage, reward_lm, size, total_reward = 0, 0, 0, 0
  local batchPieces = batch:splitIntoPieces(opt.max_bptt)
  for j = 1, batchPieces do
      local y_rf = self:forwardSample(batch, encoderStates, context, greedy)

      self.reward:resize(y_rf:size()):zero()
      self.reward_mask:resize(y_rf:size()):zero()
      -- reinforce criterion in Action

      -- * Content coverage reward
    local ori_sents
    if self.args.warType == 8 then
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(batch.seqLevelBlock),
                                    batch:getTargetAlnLabels(batch.seqLevelBlock),
                                    batch:getTargetOutput(), batch:getTargetAlnLabels(),
                                    y_rf, self.reward, self.reward_mask)
    else
      if self.args.blockAlns then
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(batch.seqLevelBlock),
                                    batch:getTargetAlnLabels(batch.seqLevelBlock), y_rf, self.reward, self.reward_mask)
      else --uses the whole document as target alns
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(),
                                    batch:getTargetAlnLabels(), y_rf, self.reward, self.reward_mask)
      end
    end

      reward_coverage = reward_coverage + self.reward:sum()

      -- combine all rewards together into self.reward
      for i = 1, self.reward:size(1) do
        for j = self.reward:size(2), 1, -1 do
          if self.reward[{ i, j }] ~= 0 then
            self.reward[{ i, j }] = self.reward[{ i, j }] * self.args.waWeight
            break
          end
        end
      end

      size = size + self.reward:sum(2):ne(0):sum()
      total_reward = total_reward + self.reward:sum()

      batch:nextPiece()
  end
  return total_reward, size, reward_coverage, reward_lm
end


--[[forwards decoder on training sequences and returns attention scores (or distribution)
-- for evaluation and visualisation of learnt alignments.]]
function ReinforceDecoder:attnScores(batch, encoderStates, context)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local f = onmt.utils.Cuda.convert(nn.SoftMax())
  local attentionMatrix = {}

  self:forwardAndApply(batch, encoderStates, context, function (outattn)
    --make attn distribution
    local attnDist = f:forward(outattn[#outattn])
    --why need to copy this?
    --just inserting outattn[#outattn] seemed to only insert references and at the end
    --all tensors of the table would be the same one. :( :(
    table.insert(attentionMatrix, onmt.utils.Cuda.convert(torch.Tensor(attnDist:size()):copy(attnDist))) --if resturn attention scores this are the last item in the output table

  end)
  return attentionMatrix

end

--[[ Compute the score of a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.

--]]
function ReinforceDecoder:computeScore(batch, encoderStates, context)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local score = {}

  self:forwardAndApply(batch, encoderStates, context, function (out, t)
    --local pred = self.generator:forward(out)
    local pred = self.generatorClones[1]:forward(out)
    for b = 1, batch.size do
      if t <= batch.targetSize[b] then
        score[b] = (score[b] or 0) + pred[1][b][batch.targetOutput[t][b]]
      end
    end
  end)

  return score
end

function ReinforceDecoder:greedyFixedFwd(batch, encoderStates, context, probBuf)
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
      local preds = self.generatorClones[1]:forward(prevOut)
      torch.max(self.maxes, self.argmaxes, preds[1], 2)
      if probBuf then
          probBuf[t]:copy(self.maxes:view(-1))
      end
      self.greedy_inp[t+1]:copy(self.argmaxes:view(-1))
    end
    return self.greedy_inp
end
