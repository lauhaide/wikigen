--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 24/02/17
-- Time: 10:07
-- To change this template use File | Settings | File Templates.
--

if opt.use_cuda then
  require 'cunn'
  require 'cutorch'
end

if opt.use_cuda then
  Tensor = torch.CudaTensor
else
  Tensor = torch.Tensor
end

localize = function(thing)
  if opt.use_cuda then
    return thing:cuda()
  end
  return thing
end


function save_parameters(scorer, filename)
  local weights, gradients = scorer:getParameters()
  if opt.use_cuda then
    weights = weights:double()
  end
  torch.save(filename, weights)
end

function load_parameters(model, filename)

  local fweights, fgradients = model:getParameters()
  print("model schema:")
  print(fweights:size())
  print("saved weights:")
  print(torch.load(filename):size())
  if  opt.use_cuda then
    fweights:copy(torch.load(filename):cuda())
  else
    fweights:copy(torch.load(filename))
  end
end

function load_vectors(filename)
  if  opt.use_cuda then
     return torch.load(filename):cuda()
  else
     return torch.load(filename)
  end
end