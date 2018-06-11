--[[ Default decoder generator. Given RNN state, produce categorical distribution.

Simply implements $$softmax(W h + b)$$.
--]]

local ReinforceGenerator, parent = torch.class('onmt.ReinforceGenerator', 'nn.Container')


function ReinforceGenerator:__init(rnnSize, outputSize)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize)
  self.outputSize = outputSize
  self:add(self.net)
end

function ReinforceGenerator:_buildGenerator(rnnSize, outputSize)
    print("* RL generator created")

    local decoderOutput = nn.Identity()()
    local y_a = nn.Linear(rnnSize, outputSize)(decoderOutput)
    local y_prob = nn.LogSoftMax()(y_a):annotate{name = 'y_softmax'}
    local out_sample = nn.ReinforceSampler('multinomial', false)(y_prob)

    return localize(nn.gModule({decoderOutput},{y_prob, out_sample}))

end


function ReinforceGenerator:updateOutput(input)
  self.output =  self.net:updateOutput(input)
  return self.output
end

function ReinforceGenerator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function ReinforceGenerator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end

--[[ Move the network to train mode. ]]
function ReinforceGenerator:training()
  --parent.training(self)
    self.net:training()
end

--[[ Move the network to evaluation mode. ]]
function ReinforceGenerator:evaluate()
    self.net:evaluate()
end
