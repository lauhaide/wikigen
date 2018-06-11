--[[ Multi-Task with Alignment Labels prediction decoder generator.

--]]
local GuidedAlnGenerator, parent = torch.class('onmt.GuidedAlnGenerator', 'nn.Container')


function GuidedAlnGenerator:__init(rnnSize, outputSize)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize)
  self:add(self.net)
end

function GuidedAlnGenerator:_buildGenerator(rnnSize, outputSize)
    print("Alignment prediction generator.")
    local wordGenerator = nn.Sequential()
        :add(nn.Linear(rnnSize, outputSize))
        :add(nn.LogSoftMax())

    --local alnClassifier =  nn.Sequential()
    --    :add(nn.Linear(rnnSize, 1))
    --    :add(nn.Sigmoid())

    local alnClassifier = nn.Sequential()
        :add(nn.Linear(rnnSize, 3))
        :add(nn.LogSoftMax())


    return nn.ConcatTable()
        :add(wordGenerator)
        :add(alnClassifier)

end

function GuidedAlnGenerator:updateOutput(input)
  --self.output = {self.net:updateOutput(input) }
  self.output =  self.net:updateOutput(input)
  return self.output
end

function GuidedAlnGenerator:updateGradInput(input, gradOutput)
  --self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function GuidedAlnGenerator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
