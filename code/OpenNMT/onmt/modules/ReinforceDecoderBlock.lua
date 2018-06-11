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


local ReinforceDecoderBlock, parent = torch.class('onmt.ReinforceDecoderBlock', 'onmt.ReinforceDecoder')

function ReinforceDecoderBlock:setSingleBlock(bnumber)
  self.enabledBlock = bnumber
end


function ReinforceDecoderBlock:setSampleStart(start)
  --update only if the start sampling position is not lower than a minumum sequence position
  if start >= self.args.minimumSamplingPos then
    self.sampleStart = start
  else
    self.sampleStart = self.args.minimumSamplingPos
  end
end


function ReinforceDecoderBlock:getSampleStart()
  return self.sampleStart
end


--[[Compute all forward steps with sampling for RL evaluation

  Parameters:

  * `batch` - `Batch` object
  * `encoderStates` -
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function ReinforceDecoderBlock:forwardSample(batch, encoderStates, context, greedy)
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
  local maxTy = self.sampleStart > Ty and Ty or Ty_rf
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


--[[Compute all forward steps with sampling for RL training

  Parameters:

  * `batch` - `Batch` object
  * `encoderStates` -
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function ReinforceDecoderBlock:forwardAndApplyRL(batch, encoderStates, context, func)
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
  if ( self.sampleStart > Ty or self.sampleStart > Ty_rf ) then
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

    --local pred, out_sample = unpack(self.generator:forward(prevOut))
    pred, out_sample = unpack(self.generatorClones[t]:forward(prevOut))

    --decide whether to start using sampled next inputs with in a batch block
    if t < self.sampleStart or not redSeq then
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
function ReinforceDecoderBlock:forward(batch, encoderStates, context)
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
  self.rf_criterion:setSampleStart(self.sampleStart)

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
function ReinforceDecoderBlock:backward(batch, allOutputs, criterion, ctxLen)

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
  if ( self.sampleStart > Ty or self.sampleStart > Ty_rf ) then
    maxTy =  Ty
    redSeq = false
  else
    maxTy =  Ty_rf
    redSeq = true
  end

  -- RL reward stuff, which needs a complete forward pass including predictions and actual output y sequence
  local rf_loss, n_seq, df_y_rf, df_expected_rewards
  local end_pos = maxTy

  if self.sampleStart <= maxTy and self.sampleStart <= Ty_rf then

    --at which possition does nll training ends
    end_pos = self.sampleStart - 1

    self.reward:resize(y_rf:size()):zero()
    self.reward_mask:resize(y_rf:size()):zero()
    -- reinforce criterion in Action

    -- * Content coverage reward
    local ori_sents, reward2
    if self.args.warType == 7 then
        _, reward2, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(batch.seqLevelBlock),
                                    batch:getTargetAlnLabels(batch.seqLevelBlock),
                                    batch:getTargetOutput(), batch:getTargetAlnLabels(),
                                    y_rf, self.reward, self.reward_mask, self.sampleStart)
    elseif self.args.warType == 8 or self.args.warType == 9 then
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(batch.seqLevelBlock),
                                    batch:getTargetAlnLabels(batch.seqLevelBlock),
                                    batch:getTargetOutput(), batch:getTargetAlnLabels(),
                                    y_rf, self.reward, self.reward_mask, self.sampleStart)
    else
      if self.args.blockAlns then
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(batch.seqLevelBlock),
                                    batch:getTargetAlnLabels(batch.seqLevelBlock), y_rf, self.reward, self.reward_mask, self.sampleStart)
      else --uses the whole document as target alns
        _, _, ori_sents = self.wa_scorer:getDynBatch(batch:getTargetOutput(),
                                    batch:getTargetAlnLabels(), y_rf, self.reward, self.reward_mask, self.reward_mask, self.sampleStart)
      end
    end

    -- combine all rewards together into self.reward
    for i = 1, self.reward:size(1) do
      for j = self.reward:size(2), 1, -1 do
        if self.reward[{ i, j }] ~= 0 then
          if self.args.warType == 7 then
            self.reward[{ i, j }] = self.reward[{ i, j }] * self.args.waWeight + reward2[{ i, j }] * self.args.lmWeight
          else
            self.reward[{ i, j }] = self.reward[{ i, j }] * self.args.waWeight
          end
          break
        end
      end
    end
  -- ** --

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
    if t < self.sampleStart or not redSeq then
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

    local df_y_rf_t = df_y_rf[{ {}, t+1 }]
    df_y_rf_t = df_y_rf_t:contiguous():view(df_y_rf_t:size(1),1)

    -- Compute the final layer gradient.
    -- Take from the y_rf gradient the part that corresponds to the targetOutput sequence

    local decGradOut = self.generatorClones[t]:backward(outputs[t], {genGradOut[1], df_y_rf_t}) --df_y_rf is already batch normalised

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

