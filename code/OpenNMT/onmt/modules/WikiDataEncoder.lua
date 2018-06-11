package.path = package.path .. ';../data2text-align/dta/?.lua'
local dbEncoder  = require('dbEncoder')
local embeddingl = require('embeddingl')
require 'rnn'

local WikiDataEncoder, parent = torch.class('onmt.WikiDataEncoder', 'nn.Container')

function WikiDataEncoder:__init(args)
    parent.__init(self)
    self.args = args
    self.network = self:_buildModel()
    self:add(self.network)
end


--[[Copy encoder weights from those in a no-alignment encoder-decoder model ]]
function WikiDataEncoder:copyFrom(tmpModelEncoder)
    local p, _ = self.network:getParameters()
    local tmpP, _ = tmpModelEncoder.network:getParameters()
    p:copy(tmpP)
end


--[[DTA weights include embedding layer and property-value LSTM.
-- The embedding layer might be with a slightly different vocabulary so
-- indexes and sizes might not match.
-- These should be copied word-by-word]]
function WikiDataEncoder:copyDTAWeights(opt, dtaModel, dtaDataDict, dataDict)
    print("Embeddings weights before: ", self.network:getParameters():sum())
    local pRnn, netEmb, dtaEmb
    for _, node2 in ipairs(self.network.forwardnodes) do
        if node2.data.annotations.name == 'DataEncoder' then
            netEmb = node2.data.module.modules[1].modules[1].weight --this is the matrix!
             --we initialise the biLSTM encoder module, the 1st module is the embedding matrix depending on vocabulary size
            pRnn = node2.data.module.modules[2]:getParameters()
        end
    end
    for _, node in ipairs(dtaModel.forwardnodes) do
        if node.data.annotations.name == 'DataEncoder' then
            dtaEmb = node.data.module.modules[1].modules[1].weight --this is the matrix!
            pRnn:copy(node.data.module.modules[2]:getParameters())
        end
    end
    print("Embeddings weights midle: ", self.network:getParameters():sum())
    embeddingl.copyMaskedLookupTables(netEmb, dtaEmb, dataDict, dtaDataDict)
    print("Embeddings weights after: ", self.network:getParameters():sum())

end

--[[ Return data to serialize. ]]
function WikiDataEncoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end


function WikiDataEncoder:_buildModel()
    local args = self.args

    local dbp = nn.Identity()() --format??
    local enc = dbEncoder.getDataEncoderOneSequenceBiLSTM_2D(args, args.vocabSize, args.train) --todo make this variable with all enc options
    local context = enc(dbp):annotate{name = 'DataEncoder', description = 'knowledge base encoder'}


    local ctx = context

    --[[FORMAT FOR DECODER INITIALISATION. Aggregate the input representation to be used to initialise decoder 0 state ]]

    local flattenedByRows = nn.Mean(2)(context)

    -- finally need to make something that can be copied into an lstm
    self.transforms = {}
    local outputs = {}
    for i = 1, args.effectiveDecLayers do
        --local lin = nn.Linear(args.nRows*args.encDim, args.decDim)
        local lin = nn.Linear(args.encDim, args.decDim)
        table.insert(self.transforms, lin)
        table.insert(outputs,
          args.tanhOutput and nn.Tanh()(lin(flattenedByRows)) or lin(flattenedByRows))
    end

    table.insert(outputs, ctx)
    local mod = nn.gModule({dbp}, outputs)
    -- output is a table with an encoding for each layer of the dec, followed by the ctx
    return localize(mod)
end

function WikiDataEncoder:shareTranforms()
    for i = 3, #self.transforms do
        if i % 2 == 1 then
            self.transforms[i]:share(self.transforms[1], 'weight', 'gradWeight', 'bias', 'gradBias')
        else
            self.transforms[i]:share(self.transforms[2], 'weight', 'gradWeight', 'bias', 'gradBias')
        end
    end
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states: layer-length table with batchSize x decDim tensors
  2. - context matrix H: batchSize x nRows*nCols x encDim
--]]
function WikiDataEncoder:forward(batch)
  local finalStates = self.network:forward(batch:getSource())
  local context = table.remove(finalStates) -- pops, i think
  return finalStates, context
end

--[[ Backward pass (only called during training)

  Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state
  * `gradContextOutput` - gradient of loss wrt full context.

  Returns: `gradInputs` of input network.
--]]
function WikiDataEncoder:backward(batch, gradStatesOutput, gradContextOutput)
    local encGradOut = {}
    for i = 1, self.args.effectiveDecLayers do -- ignore input feed (and attn outputs)
        table.insert(encGradOut, gradStatesOutput[i])
    end
    table.insert(encGradOut, gradContextOutput)
    local gradInputs = self.network:backward(batch:getSource(), encGradOut)
    return gradInputs
end

function WikiDataEncoder:loadDTAWeights(dtaModel)

end