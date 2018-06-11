package.path = './onmt/reinforce_components/?.lua;' .. package.path
package.path = '../dress/dress/dress/nnets/?.lua;' .. package.path
package.path = '../dress/dress/dress/utils/?.lua;' .. package.path
package.path = '../dress/dress/dress/layers/?.lua;' .. package.path
package.path = '../dress/dress/dress/dataset/?.lua;' .. package.path
package.path = '../dress/dress/dress/reinforce_components/?.lua;' .. package.path

package.path = package.path .. ';../data2text-align/?.lua'
require('dta.dbEncoder')
require('dta.milTriplesSM2Deep')

package.path = package.path .. ';../data2text-align/?.lua'
require 'evaluationUtils'

require('onmt.init')

local path = require('pl.path')
tds = require('tds')
local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data_file','', [[Path to the training *.hdf5 file from Preprocess.py]])
cmd:option('-valid_data_file','', [[Path to validation *.hdf5 file from Preprocess.py]])
cmd:option('-test_data_file','', [[Path to test *.hdf5 file from Preprocess.py]])
cmd:option('-data_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-text_dict', '', [[Path to target vocabulary (*.targ.dict file)]])

-- ** from my generator ** --
cmd:option('-dataEncoder', 8, [[Type of encoding for triple/property data.]])
cmd:option('-use_cuda', true, [[Use GPU]])
cmd:option('-preembed', false, [[Initialise embedding layers with pre-trained vectors]])
cmd:option('-preembed_datavoc', '', [[File with pre-trained vectors for data vocabulary dict.]])
cmd:option('-hdec', false, [[]])
cmd:option('-waEvaluationSet', '',
    [[Name (with path) of text file containing selected item ids for word alignment based evaluation]])
cmd:option('-evalAln', false, [[]])
cmd:option('-maskNoneField', false, [[Evaluate decoder predictions using input property set without NONEFIELD.]])
cmd:option('-dtaWeights', '', [[]])
cmd:option('-dataDictDTA', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-textDictDTA', '', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-decodingMaxLength', 540, [[Type of encoding for triple/property data.]])
cmd:option('-test', false, [[]])


cmd:text("")
cmd:text("**Alignment prediction multi-task**")
cmd:text("")
cmd:option('-guided_alignment', 0,'[[If 1, use word alignment labels created with DTA-aligner model for combined objective training]]')
cmd:option('-guided_alignment_weight', 0.5, [[default weights for external alignments objective]])
cmd:option('-lm_objective_decay', 0, [[decay rate per epoch for LM weight - typical with 0.9,
                                         weight will end up at ~30% of its initial value]])
cmd:option('-start_guided_decay', 7,'[[If 1, use word alignment labels created with DTA-aligner model for combined objective training]]')


cmd:text("")
cmd:text("**Reinforce options**")
cmd:text("")

cmd:option('-reinforce', false, [[]])
cmd:option('-lmWeight', 0.5, [[Weigth for the LM based reward.]])
cmd:option('-waWeight', 1, [[Weigth for the Word Alignment based reward.]])
cmd:option('-lmPath', '', [[Path to pre-trained Language Model.]])
cmd:option('-embPath', '', [[Path to pre-trained DTA embeddings.]])
cmd:option('-deltaSamplePos', 5, 'delta')
cmd:option('-initialOffset', 5, 'Initial sample start offset')
cmd:option('-rfEpoch', 2, 'num of epochs for each rl phrase')
cmd:option('--enableSamplingFrom', 20, [[postition from which RL can start sampling.
                                When training at document level this goes for the whole sequence.
                                When training at block level this is valid only for first block, other ones decrease up to 1st position.]])
cmd:option('-structuredCopy', false, [[]])
cmd:option('-blockRL', false, [[Whether RL training occurs through blocks in a document (false) or within blocks (true). ]])
cmd:option('-blockWaReward', false, [[Whether the reward uses the alignments of blocks (true) or whole document (false).
For certain reward types this is ignored (e.g. warType=8). ]])
cmd:option('-warType', 0, [[1. Bleu1 2.Fmeasure(Bleu1, recall) 3.weighted Bleu1 + weighted recall 4.history of rews(decrease 1 to mentioned words)]])

-- ** ** --



cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]])
cmd:option('-just_eval', false, [[]])
cmd:option('-just_gen', false, [[]])
cmd:option('-beam_size', 5, [[]])
cmd:option('-gen_file', 'preds.txt', [[]])
cmd:option('-verbose_eval', false, [[]])
cmd:option('-scoresomethings', false, [[]])
cmd:option('-unkfilter', 3, [[]])


cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 200, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 200, [[Word embedding sizes]])
cmd:option('-feat_merge', 'concat', [[Merge action for the features embeddings: concat or sum]])
cmd:option('-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
cmd:option('-residual', false, [[Add residual connections between RNN layers.]])
cmd:option('-just_lm', false, [[No conditioning]])
cmd:option('-copy_generate', false, [[]])
cmd:option('-tanh_query', false, [[]])
cmd:option('-poe', false, [[]])
cmd:option('-recdist', 0, [[Distance to use if doin continuous reconstruction]])
cmd:option('-discrec', false, [[Do discrete reconstruction]])
cmd:option('-discdist', 0, [[1 for total dev; 2 for hellinger]])
cmd:option('-recembsize', 300, [[Embedding size of entries to reconstruct]])
cmd:option('-partition_feats', false, [[Partition feats used in discrete reconstruction]])
cmd:option('-nfilters', 200, [[Convolutional filters for reconstruction]])
cmd:option('-nrecpreds', 3, [[Number of entries to reconstruct]])
cmd:option('-rho', 0.5, [[Reconstruction loss coefficient]])

cmd:option('-switch', false, [[]])
cmd:option('-multilabel', false, [[]])
cmd:option('-map', false, [[]])

cmd:option('-pool', 'mean', [[mean or max]])
cmd:option('-enc_layers', 1, [[]])
cmd:option('-enc_emb_size', 200, [[]])
cmd:option('-enc_dropout', 0, [[]])
cmd:option('-enc_relu', false, [[]])
cmd:option('-enc_tanh_output', false, [[]])
cmd:option('-double_output', false, [[]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-start_iteration', 1, [[If loading from a checkpoint, the iteration from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd, adagrad, adadelta, adam, mom]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings are: sgd = 1,
                                adagrad = 0.1, adadelta = 1, adam = 0.0002]])
cmd:option('-mom', 0.9, [[momentum]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-learning_rate_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 10000, [[Start decay after this epoch]])
cmd:option('-decay_update2', false, [[Decay less]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]])
cmd:option('-max_bptt', 500, [[]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', 0, [[1-based identifier of the GPU to use. CPU is used when the option is < 1]])
cmd:option('-nparallel', 1, [[When using GPUs, how many batches to execute in parallel.
                            Note: this will technically change the final batch size to max_batch_size*nparallel.]])
cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])

-- bookkeeping
cmd:option('-save_every', 0, [[Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. ]])
cmd:option('-report_every', 50, [[Print stats every this many iterations within an epoch.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-json_log', false, [[Outputs logs in JSON format.]])

opt = cmd:parse(arg)
package.path = '../data2text-align/utils/?.lua;' .. package.path
require 'cuda'

--move this to wiki constants
ALIGNID = 2
NOALIGNID = 3

function configDetails()
  local options = ""
  for k, v in pairs(opt) do
      options = options .. "\n" .. k ..": " .. tostring(v) .. " "
  end
  return options
end

local function reseed()
  torch.manualSeed(opt.seed)
  if opt.gpuid > 0 then
    cutorch.manualSeed(opt.seed)
  end
end

local function structuredParametersLoad(checkpoint)
    local tmpOpt = checkpoint.options
    local tmpModel = {}
    local verbose = true
    local tripV

    if tmpOpt.hdec then
        -- make sentence decoder first
        local sentDecoder =  onmt.Models.buildDecoder(tmpOpt, checkpoint.dicts.tgt, verbose, tripV)
        tmpModel.decoder = onmt.Models.buildHirarchicalDecoder(tmpOpt, verbose, sentDecoder)
    else
        -- make decoder first
        tmpModel.decoder = onmt.Models.buildDecoder(tmpOpt, checkpoint.dicts.tgt, verbose, tripV)
    end

    -- send to gpu immediately to make cloning things simpler
    onmt.utils.Cuda.convert(tmpModel.decoder)

        --encoders will be always the same
        tmpModel.encoder = onmt.WikiDataEncoder({
        vocabSize = checkpoint.dicts.src.words:size(),
        encDim = tmpOpt.enc_emb_size,
        decDim = tmpOpt.rnn_size,
        nLayers = tmpOpt.enc_layers,
        pool = tmpOpt.pool or "mean",
        effectiveDecLayers = tmpOpt.layers*2,
        dropout = tmpOpt.enc_dropout,
        relu = tmpOpt.enc_relu,
        wordVecSize = tmpOpt.word_vec_size,
        tanhOutput = tmpOpt.enc_tanh_output,
        input_feed = tmpOpt.input_feed,
        rnn_size = tmpOpt.rnn_size,
        preembed = tmpOpt.preembed,
        word_vec_size = tmpOpt.word_vec_size,
        preembed_datavoc = tmpOpt.preembed_datavoc,
        train = 1
    })
    onmt.utils.Cuda.convert(tmpModel.encoder)
    local everything = nn.Sequential()
    for k, mod in pairs(tmpModel) do
        everything:add(mod)
    end

    local p, gp = everything:getParameters()
    print(p:size())
    print(checkpoint.flatParams[1]:size(1))
    p:copy(checkpoint.flatParams[1])
    print("Structured parameters happily copied !!")
    return tmpModel
end

local function intiFromDTA(opt, model, dataset)

    -- load dta with weigths from file .t7 voc_size and text_size are from original dicst used for the encoder
    -- -text_dict and  -data_dict are needed. Further below copyDTAWeights() will also need the four dictionaries
    -- to convert indices and initialise embeddings

    print('loading dicts ...')
    --load dictionaries for key2idx search
    local dataDictDTA, vocDataSize = key2idx(opt.dataDictDTA)
    local textDictDTA, vocTextSize = key2idx(opt.textDictDTA)
    local dtaModel = load_alignment_scorer(optDTA, vocDataSize, vocTextSize, dataset, opt.dtaWeights)

    -- initialise weights of the whole encoder
    model.encoder:copyDTAWeights(opt, dtaModel, dataDictDTA, idx2word_data)
    -- initialise weights of decoder embedding layer
    model.decoder:copyDTAWeights(dtaModel, textDictDTA, idx2word_text)

end

local function initParams(model, verbose, dataset)
    local numParams = 0
    local params = {}
    local gradParams = {}

    if verbose then
        print('Initializing parameters...')
    end

    -- we assume all the sharing has already been done,
    -- so we just make a big container to flatten everything
    local everything = nn.Sequential()
    for k, mod in pairs(model) do
        print(k,mod:getParameters():size(1))
        everything:add(mod)
    end

    local p, gp = everything:getParameters()


    if opt.train_from:len() == 0 then
        p:uniform(-opt.param_init, opt.param_init)
        -- do module specific init; wordembeddings will happen multiple times,
        -- but who cares
        for k, mod in pairs(model) do
            mod:apply(function (m)
                if m.postParametersInitialization then
                    m:postParametersInitialization()
                end
            end)
        end

        if opt.dtaWeights:len() ~= 0 then
            intiFromDTA(opt, model, dataset)

            local everything = nn.Sequential()
            for k, mod in pairs(model) do
              everything:add(mod)
            end

            p, gp = everything:getParameters()
        end
    else
        print("copying loaded params...")
        local checkpoint = torch.load(opt.train_from)

        if opt.structuredCopy then
            local tmpModel = structuredParametersLoad(checkpoint)

            model.encoder:copyFrom(tmpModel.encoder)
            model.decoder:copyFrom(tmpModel.decoder)

            for k, mod in pairs(model) do
                mod:apply(function (m)
                if m.postParametersInitialization then
                    m:postParametersInitialization()
                end
                end)
            end

            local everything = nn.Sequential()
            for k, mod in pairs(model) do
              everything:add(mod)
            end

            p, gp = everything:getParameters()

          --[[
          -- This are some parts which could be shared across different model variants.
          -- I.e. if we want to recover parameters from a different model variant, then
          -- we need to copy only parts in common. E.g. if we load from plain encoder-decoder
          -- into reinforce encoder-decoder we need to copy same subnets becuase there are
          -- model components that might be different from one to the other.
          -- this means we cannot just copy fattenes parameters.
          -- Solution: either create a source model initialise with its flat parameters and then
          -- access and copy the necessary parts. Otherwise, save the model  (Checkpoint.lua) in structured way rather
          -- than flat parameters.
          print(self.model.decoder.rnn)
          print(self.model.decoder.inputNet)
          print(self.model.decoder.generator.net.forwardnodes[3].data.module)
           ]]
        else
            p:copy(checkpoint.flatParams[1])

        end

    end

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)

    if verbose then
        print(" * number of parameters: " .. numParams)
    end
    return params, gradParams
end

--[[
-- Guided alignment criterion can combine with other simple or copy_generate criterions.
--]]
local function buildCriterion(vocabSize, features)
  local criterion = nn.ParallelCriterion(false)

  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    if opt.guided_alignment == 1 then
        criterion:add(nll, 1 - opt.guided_alignment_weight)
    else
        criterion:add(nll)
    end
  end

  if opt.copy_generate then
       local marginalCrit = onmt.MarginalNLLCriterion(onmt.Constants.PAD)
       marginalCrit.sizeAverage = false
       if opt.guided_alignment == 1 then
         criterion:add(marginalCrit, 1 - opt.guided_alignment_weight)
       else
         criterion:add(marginalCrit)
       end
  else
      addNllCriterion(vocabSize)
  end

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end

  if opt.guided_alignment == 1 then
      print("Alignment prediction criterion.")
      local biw = torch.ones(3) --YES, NO class plus "1" padding of the sequence
      biw[onmt.Constants.PAD] = 0
      local bicriterion = nn.ClassNLLCriterion(biw)
      bicriterion.sizeAverage = false
      criterion:add(bicriterion, opt.guided_alignment_weight)
      --This worked for me because I was using masking.
      --here different seq.lengths and padding is implemented differently.
      --Then padding symbol overlaps with BCE 0/1 targets. Need to do sthg.
      --local bicriterion = nn.BCECriterion()
      --bicriterion.sizeAverage = false
      --bicriterion = localize(nn.MaskZeroCriterion(bicriterion,1))
      --criterion:add(bicriterion, opt.guided_alignment_weight)
  end

  return criterion
end

function allTraining(model)
    for _, mod in pairs(model) do
        if mod.training then
            mod:training()
        end
    end
end

function allEvaluate(model)
    for _, mod in pairs(model) do
        if mod.evaluate then
            mod:evaluate()
        end
    end
end

-- gets encodings for all rows
function allEncForward(model, batch)
    local aggEncStates, catCtx = model.encoder:forward(batch)
    if opt.just_lm then
        for i = 1, #aggEncStates do
            aggEncStates[i]:zero()
        end
        catCtx:zero()
    end
    return aggEncStates, catCtx
end

-- goes backward over all encoders
function allEncBackward(model, batch, encGradStatesOut, gradContext)
    model.encoder:backward(batch, encGradStatesOut, gradContext)
end

local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  allEvaluate(model)
  for i = 1, data:batchCount() do
    model.decoder:resetLastStates()
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local aggEncStates, catCtx = allEncForward(model, batch)
    loss = loss + model.decoder:computeLoss(batch, aggEncStates, catCtx, criterion)
    total = total + batch.targetNonZeros
  end
  allTraining(model)

  return math.exp(loss / total)
end



local function evalSample(model, criterion, data, greedy)
  local totalCnt = 0
  local totalLoss = 0
  local cnt = 0
  local coverageLoss, lmLoss = 0, 0

  allEvaluate(model)
  for i = 1, data:batchCount() do
    model.decoder:resetLastStates()
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local aggEncStates, catCtx = allEncForward(model, batch)  --- ***MERGE with eval so to reuse encoder computation
    local reward, size, r_coverage, r_lm  = model.decoder:computeSampleLoss(batch, aggEncStates, catCtx, greedy)

    totalLoss = totalLoss + reward
    totalCnt = totalCnt + size

    coverageLoss = coverageLoss + r_coverage
    lmLoss = lmLoss + r_lm

    cnt = cnt + 1
    if cnt % 5 == 0 then
      collectgarbage()
    end
  end

  return totalLoss / totalCnt, coverageLoss / totalCnt, lmLoss / totalCnt
end


--[[Generates output for the evaluation of word alignments against the manually
--annotated ones (Yawat files).
-- ]]
local function evalAligner(model, dataset)
  local START = 3
  local END = 4

  --[[get selected examples for word alignment -based evaluation]]
  local yawatEvalSet = readYawatEvalSelection(opt.waEvaluationSet)
  print("read item selection for wa-based evaluation...")

  local waMatrices = {}
  local cntYawatCases = 0

  allEvaluate(model)
  for i = 1, dataset:batchCount() do
    model.decoder:resetLastStates()
    local batch = onmt.utils.Cuda.convert(dataset:getBatch(i))
    local aggEncStates, catCtx = allEncForward(model, batch)
    local attn_coefs = model.decoder:attnScores(batch, aggEncStates, catCtx)

    attn_coefs =
        nn.View(batch.size, batch.sourceLength, batch.targetLength)
            :forward(nn.JoinTable(2):forward(attn_coefs))

    for k = 1, batch.size do
        if yawatEvalSet:eq(batch:getOrigIDs()[k][1]):sum() > 0 then
            input_tokens = batch:getSource()[k]
            dataSeqL = input_tokens:size()[1]
            textSeqL = batch:getTargetInput()[k]:size()[1]
            input_set = localize(nn.Replicate(1,1)):forward(input_tokens)

            -- format alignment matrix
            local datasing = dataset:getSingleDataSequence(input_set, 1, opt.dataEncoder) --returns right aligned sequence
            local alignment, dataseq = adjustMatrix2Sequence(
                {datasing, batch:getTargetOutput()[k]}, dataset.max_prop_l, dataset.max_dataseq_l, attn_coefs[k], g_targetDictEd, RPADDING)
            local seqFragments = sequenceFragScores({datasing, batch:getTargetOutput()[k]}, attn_coefs[k], RPADDING)
            table.insert(waMatrices, {batch:getOrigIDs()[k], alignment, input_set[1], seqFragments, 0.0,
                batch:getTargetOutput()[k][batch:getTargetOutput()[k]:gt(1)]})
            cntYawatCases = cntYawatCases +1
        end
    end
  end
  evalAlignment(waMatrices, g_targetDictEd, opt.gen_file, false) -- opt.gen_file here should be the output directory
  print("nb. of generated files:", cntYawatCases)
end

local function convert_and_shorten_string(ts, max_len, dict)
   local strtbl = {}
   for i = 1, max_len do
       if ts[i] == onmt.Constants.EOS then
           break
       end
       table.insert(strtbl, dict.idxToLabel[ts[i]])
   end
   return stringx.join(' ', strtbl)
end

function seqToText(dict, seq)
    --[[for debuggin purpose]]
    local text = ""
    for j=1, seq:size()[1] do

        if dict[seq[j]] ~= nill then
            text = text .. dict[seq[j]] .. " "
        elseif seq[j]==0 then
            text = text .. " 0 "
        else
            print(string.format("no dict entry for index %d", seq[j]))
        end
    end
    return text
end

function convertSourceData(propertySet, dict)
    local inputTokens = {}
    for i=1, propertySet:size(1) do
        if propertySet[i][propertySet[i]:gt(0)]:sum() > 0 then
          local strprop =  seqToText(dict, propertySet[i][propertySet[i]:gt(0)])
          table.insert(inputTokens, strprop)
        end
    end
    return table.concat(inputTokens, " | ")
end

function tensor2oTable(v)
    t = {}
    for i=1,v:size(1) do
      t[i] = v[i]
    end
    return t
end

local function beamGen(model, data, tgtDict, srcDict)
  -- adapted from Translator:translateBatch()
  local nbMBatches = data:batchCount()
  local doBatches
  if opt.test then
    doBatches = torch.range(1,nbMBatches)
  else
    doBatches = torch.cat(torch.cat(torch.range(1,100),torch.range(nbMBatches-100,nbMBatches)),
        torch.range((math.ceil(nbMBatches/2))-100,(math.ceil(nbMBatches/2))+100),1)
  end

  local  maxBatches
  if nbMBatches >= doBatches:size(1) then
      maxBatches = doBatches
  else
     maxBatches = torch.range(1, nbMBatches)
  end
  print(" * For evaluation doing ", maxBatches:size(1), "batches")



  --local max_sent_length = 1500
  local max_sent_length = opt.decodingMaxLength
  print("using max len:", max_sent_length)
  allEvaluate(model)
  local ibatch, iCurids, srcs, tgtOut

  local outFile = io.open(opt.gen_file, 'w')

  for i = 1, maxBatches:size(1) do
    model.decoder:resetLastStates()
    --ibatch = data:getBatch(i)
    print("get batch id ", i, maxBatches[i])
    ibatch = data:getBatch(maxBatches[i])
    iCurids = ibatch:getOrigIDs()
    srcs = ibatch:getSource()
    tgtOut = ibatch:getTargetOutput()
    local batch = onmt.utils.Cuda.convert(ibatch)

    local aggEncStates, catCtx = allEncForward(model, batch)
    local advancer
    if opt.switch then
        advancer = onmt.translate.SwitchingDecoderAdvancer.new(model.decoder,
           batch, catCtx, max_sent_length, nil, aggEncStates, nil, opt.map, opt.multilabel)
    else
        advancer = onmt.translate.Decoder2Advancer.new(model.decoder,
           batch, catCtx, max_sent_length, nil, aggEncStates, nil, opt)
    end
    local beamSearcher = onmt.translate.BeamSearcher.new(advancer)
    local results = beamSearcher:search(opt.beam_size, 1, 1, false)
    for b = 1, batch.size do
        local top1 = results[b][1].tokens
        local top1tostr = convert_and_shorten_string(top1, #top1, tgtDict)
        print(iCurids[b][1],top1tostr)
        local ref = tensor2oTable(tgtOut[b][tgtOut[b]:ne(1)]) --eliminate paddings and convert to table
        local targetOutSentenceRef = convert_and_shorten_string(ref, #ref, tgtDict)

        outFile:write(string.format("curid=%d.sid=%d\n", iCurids[b][1], iCurids[b][2]))
        outFile:write("src= " .. convertSourceData(srcs[b], srcDict) .. "\n")
        outFile:write("tst= " .. top1tostr .. "\n")
        outFile:write("ref1= " .. targetOutSentenceRef .. "\n")
    end
  end
  outFile:close()
end

--[[IMPLEMENTATION of this function is NOT FINISHED]]
local function trainEpochHirarchical(epoch, lastValidPpl, model, optim, trainData, crt, prms, checkpoint)
    local epochState
    local batchOrder
    local startI = opt.start_iteration
    local criterion, recCrit, switchCrit, ptrCrit = unpack(crt)
    local params, gradParams = unpack(prms)

    local numIterations = trainData:batchCount()

    if startI > 1 and info ~= nil then
        epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, info.epochStatus)
        batchOrder = info.batchOrder
    else
        epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl)
        -- Shuffle mini batch order.
        batchOrder = torch.randperm(trainData:batchCount())
    end

    --opt.start_iteration = 1
    print(string.format("running for %d batches",trainData:batchCount()))
    local iter = 1
    local totalLoss2, totalLoss3 = 0, 0
    model.decoder:remember()
    for i = startI, trainData:batchCount() do --TODO: batches are order by sizes, shuffle batches ??? as done in edTrain2.lua
        local batchIdx = epoch <= opt.curriculum and i or batchOrder[i]
        local batch =  trainData:getBatch(batchIdx)

        batch.totalSize = batch.size

        model.decoder:resetLastStates() -- don't use saved last state for new batch
            optim:zeroGrad(gradParams)
            local aggEncStates, catCtx = allEncForward(model, batch)
            local ctxLen = catCtx:size(2)

            local decOutputs = model.decoder:forward(batch, aggEncStates, catCtx)
            local encGradStatesOut, gradContext, loss, loss2, loss3 = model.decoder:backward(batch, decOutputs,
                                                                       criterion, ctxLen, recCrit,
                                                                        switchCrit, ptrCrit)
            allEncBackward(model, batch, encGradStatesOut, gradContext)

            -- Update the parameters.
            optim:prepareGrad(gradParams, opt.max_grad_norm)
            optim:updateParams(params, gradParams)
            --epochState:update(batch, loss, recloss)
            epochState:update(batch, loss, nil)
            if loss2 then
                totalLoss2 = totalLoss2 + loss2
            end
            if loss3 then
                totalLoss3 = totalLoss3 + loss3
            end
            batch:nextPiece()

        if iter % opt.report_every == 0 then
            epochState:log(iter, opt.json_log)
            if opt.switch then
                print("switchLoss", totalLoss2/epochState.status.trainNonzeros)
                print("ptrLoss", totalLoss3/epochState.status.trainNonzeros)
            end
            collectgarbage()
        end
        if opt.save_every > 0 and iter % opt.save_every == 0 then
            checkpoint:saveIteration(iter, epochState, batchOrder, not opt.json_log)
        end
        iter = iter + 1
    end
    return epochState
end -- end local function trainEpochHirarchical


local function updateCriterionWeights(criterion)
    --weights[1] is criterion LM
    --weights[2] is criterion classif
    if opt.guided_alignment==1 and opt.lm_objective_decay==1 then
        local tmp = criterion.weights[1]
        criterion.weights[1] = criterion.weights[2]
        criterion.weights[2] = tmp
    end
    return criterion
end


local function trainModel(model, trainData, validData, testData, dataset, info)
    local criterion
    local verbose = true
    local params, gradParams = initParams(model, verbose, trainData)
    allTraining(model)

    -- define criterion of each GPU
    criterion = onmt.utils.Cuda.convert(buildCriterion(dataset.dicts.tgt.words:size(),
                                                          dataset.dicts.tgt.features))


    local recCrit
    if opt.discrec then
        recCrit = onmt.utils.Cuda.convert(nn.KMinXent())
        recCrit.sizeAverage = false
    elseif opt.recdist > 0 then
        recCrit = onmt.utils.Cuda.convert(nn.KMinDist(opt.recdist))
        recCrit.sizeAverage = false
    end

    local switchCrit, ptrCrit
    if opt.switch then
        switchCrit = onmt.utils.Cuda.convert(nn.BCECriterion())
        switchCrit.sizeAverage = false
        if opt.multilabel then
            ptrCrit = onmt.utils.Cuda.convert(nn.MarginalNLLCriterion())
            ptrCrit.sizeAverage = false
        else
            ptrCrit = onmt.utils.Cuda.convert(nn.ClassNLLCriterion())
            ptrCrit.sizeAverage = false
        end
    end

    --TODO add this next
    -- optimize memory of the first clone
    --[[
    if not opt.disable_mem_optimization then
        local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
        batch.totalSize = batch.size
        onmt.utils.Memory.boxOptimize2(model, criterion, batch, verbose, switchCrit, ptrCrit)
    end
    ]]


    local optim = onmt.train.Optim.new({
        method = opt.optim,
        numModels = 1, -- we flattened everything
        learningRate = opt.learning_rate,
        learningRateDecay = opt.learning_rate_decay,
        startDecayAt = opt.start_decay_at,
        optimStates = opt.optim_states,
        mom = opt.mom
    })


    local checkpoint = onmt.train.Checkpoint.new(opt, model, params, optim, dataset)

    local sampleStartOffset = opt.initialOffset
    local batchesDone

    local function trainEpoch(epoch, lastValidPpl)
        local epochState
        local batchOrder
        local startI = opt.start_iteration

        local numIterations = trainData:batchCount()

        if startI > 1 and info ~= nil then
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, info.epochStatus)
            batchOrder = info.batchOrder
        else
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl)
            -- Shuffle mini batch order.
            batchOrder = torch.randperm(trainData:batchCount())
        end

        print(string.format("running for %d batches",trainData:batchCount()))

        local iter = 1
        local totalLoss2, totalLoss3, totalCnt = 0, 0, 0
        model.decoder:remember()
        local skipBatch = false
        for i = startI, trainData:batchCount() do
        --for i = startI, 100 do

            local batchIdx = epoch <= opt.curriculum and i or batchOrder[i]
            local batch =  trainData:getBatch(batchIdx)
            local batchNLLTokens, batchRLTokens = 0, 0

            --Control LR curriculum learning for the whole document sequence
            --Mini batches might have different target seq lengths (due to bucketting)
            --Then different batches will reach the beginning as the start position for sampling
            --faster than others. Those that have finished should stop training.
            if opt.reinforce and not opt.blockRL then
                if ((batch.rulTargetLength - sampleStartOffset) < opt.enableSamplingFrom)  then
                    skipBatch = true
                else
                    model.decoder:setSampleStart(batch.rulTargetLength - sampleStartOffset)
                end
            end


            if not opt.reinforce or not skipBatch then

                batch.totalSize = batch.size -- fuck off

                --(mine)comment out as it's already done:
                --onmt.utils.Cuda.convert(batch)

                local skipBlock = false
                local batchPieces = batch:splitIntoPieces(opt.max_bptt)
                model.decoder:resetLastStates() -- don't use saved last state for new batch
                for j = 1, batchPieces do

                    --Control LR curriculum learning for the block level training
                    if opt.reinforce and opt.blockRL then
                        if  ((batch.targetLength - sampleStartOffset) <= 0) then
                            if j ~= 1 and (sampleStartOffset - batch.targetLength) <= opt.deltaSamplePos then
                                model.decoder:setSampleStart(1)
                            elseif j == 1 and (sampleStartOffset - batch.targetLength) <= opt.deltaSamplePos then
                                --first block will keep sampling from minimal position until curiculum goes to zero for all blocks
                                model.decoder:setSampleStart(opt.enableSamplingFrom)
                            else
                                skipBlock = true
                            end
                        elseif j == 1 and ((batch.targetLength - sampleStartOffset) <= opt.enableSamplingFrom) then
                            --first block will keep sampling from minimal position until curiculum goes to zero for all blocks
                            model.decoder:setSampleStart(opt.enableSamplingFrom)
                        else
                            model.decoder:setSampleStart(batch.targetLength - sampleStartOffset)
                        end
                    end

                    if not skipBlock then
                        optim:zeroGrad(gradParams)
                        local aggEncStates, catCtx = allEncForward(model, batch)
                        local ctxLen = catCtx:size(2)

                        local decOutputs = model.decoder:forward(batch, aggEncStates, catCtx)
                        local encGradStatesOut, gradContext, loss, loss2, loss3 = model.decoder:backward(batch, decOutputs,
                                                                                       criterion, ctxLen, recCrit,
                                                                                        switchCrit, ptrCrit)

                        allEncBackward(model, batch, encGradStatesOut, gradContext)

                        -- Update the parameters.
                        optim:prepareGrad(gradParams, opt.max_grad_norm)

                        -- Detecting and removing NaNs
                        if gradParams[1]:ne(gradParams[1]):sum() > 0 then
                           print(sys.COLORS.red .. ' warning clip weights has NaN/s')
                           NaNOk = false
                        end

                        optim:updateParams(params, gradParams)

                        -- Detecting and removing NaNs
                        if gradParams[1]:ne(gradParams[1]):sum() > 0 then
                           print(sys.COLORS.red .. ' warning update weights has NaN/s')
                           NaNOk = false
                        end

                        if opt.reinforce then
                            --epochState:update(batch, loss, recloss)

                            local nll_loss, rf_loss, nseq, nonRLNonZeros = unpack(loss)
                            epochState:updateRL(batch, nll_loss, nonRLNonZeros)
                            totalLoss2 = totalLoss2 + (rf_loss * batch.size)
                            totalCnt = totalCnt + nseq
                            --keep the following for control/debugging
                            batchNLLTokens = batchNLLTokens + nonRLNonZeros
                            batchRLTokens = batchRLTokens + nseq
                        else
                            --epochState:update(batch, loss, recloss)
                            epochState:update(batch, loss, nil)
                            if loss2 then
                                totalLoss2 = totalLoss2 + loss2
                            end
                            if loss3 then
                                totalLoss3 = totalLoss3 + loss3
                            end
                        end
                        batchesDone = batchesDone +1
                    end --if skipBlock
                    batch:nextPiece()
                end

                if batchesDone > 0 then
                    if iter % opt.report_every == 0 then
                        epochState:log(iter, opt.json_log, fd)
                        if opt.switch then
                            print("switchLoss", totalLoss2/epochState.status.trainNonzeros)
                            print("ptrLoss", totalLoss3/epochState.status.trainNonzeros)
                        elseif opt.reinforce then
                            local rlStats = string.format("          rfl: %.5f ; nseq: %d ; rl_ppl: %.5f ",
                                totalLoss2, totalCnt, totalLoss2/totalCnt)
                            print(rlStats)
                            fd:write(rlStats .. "\n")
                            assert(batchNLLTokens <= batch.rulTargetNonZeros,
                                 string.format("losses did not cover the nb of tokens... nllT: %d + rlT: %d <> %d",
                                    batchNLLTokens, batchRLTokens, batch.rulTargetNonZeros))
                        end
                        collectgarbage()
                    end
                    if opt.save_every > 0 and iter % opt.save_every == 0 then
                       checkpoint:saveIteration(iter, epochState, batchOrder, not opt.json_log)
                    end
                end
                iter = iter + 1
            end
        end
        return epochState, batchesDone == 0 --control curriculum learning for RL
    end -- end local function trainEpoch

    reseed()
    local validPpl = 0
    local bestPpl = math.huge
    local bestEpoch = -1

    if not opt.json_log then
        print('Start training... logging to')
        print(logFile)
    end

    if opt.just_gen then
        local pplLoad = eval(model, criterion, validData) --verify that the loaded params are correct
        print("PPL of trained model on valid: ", pplLoad)
        if opt.test then
            print(" * Generate on test dataset ")
            beamGen(model, testData, g_tgtDict, g_srcDict)
        else
            beamGen(model, validData, g_tgtDict, g_srcDict)
        end
        return
    elseif opt.just_eval then
        validPpl = eval(model, criterion, validData)
        if not opt.json_log then
            print('Validation perplexity: ' .. validPpl)
        end
        onmt.train.Greedy.greedy_eval(model, validData, nil, g_tgtDict, 1, 10, opt.verbose_eval)
        return
    elseif opt.evalAln then
        print("* Generate files for Yawat-based alignment evaluation")
        evalAligner(model, validData)
        return
    end

    local unupWeights = true
    local batchesCompleted = false
    for epoch = opt.start_epoch, opt.epochs do

        if opt.reinforce and batchesCompleted then
          xprintln('curriculum learning done! sample offset = %d', sampleStartOffset)
          break
        end

        if not opt.json_log then
            print('')
        end

        local epochState
        if opt.hdec then
            epochState = trainEpochHirarchical(
                epoch, validPpl, model, optim, trainData,
                {criterion, recCrit, switchCrit, ptrCrit},
                {params, gradParams}, checkpoint)
        else
            batchesDone = 0
            epochState, batchesCompleted = trainEpoch(epoch, validPpl)
            print(batchesDone, batchesCompleted)
        end

        validPpl = eval(model, criterion, validData)

        if not opt.json_log then
            local valStats = 'Validation perplexity: ' .. validPpl
            print(valStats)
            fd:write(valStats .. "\n")
        end

        --with reinforce we choose to use sgd and we do not decay learning rates
        if (opt.optim == 'sgd' or opt.optim == 'mom') and not opt.reinforce then
            if opt.decay_update2 then
                optim:updateLearningRate2(validPpl, epoch)
            else
                optim:updateLearningRate(validPpl, epoch)
            end
        end

        if validPpl < bestPpl then
            checkpoint:deleteEpoch(bestPpl, bestEpoch)
            checkpoint:saveEpoch(validPpl, epochState, not opt.json_log, timeStamp)
            bestPpl = validPpl
            bestEpoch = epoch
        elseif opt.guided_alignment then
        --save always as with changes in criterion weights there might be jumps in preplexities
            checkpoint:saveEpoch(validPpl, epochState, not opt.json_log, timeStamp)
        end
        collectgarbage()

        --TODO: better handle this when combining with other types of generators and criterions (copy, pointer etc)
        if epoch >=  opt.start_guided_decay and opt.lm_objective_decay == 1 and unupWeights then
           criterion = updateCriterionWeights(criterion)
           print(string.format("* Criterion weights switched: %f and %f", criterion.weights[1], criterion.weights[2]))
           unupWeights = false
        end

        if opt.reinforce and epoch % opt.rfEpoch == 0 then
          sampleStartOffset = sampleStartOffset + opt.deltaSamplePos
          local rlCurric = string.format('epoch %d rf offset from the end to start %d ', epoch, sampleStartOffset)
          print(rlCurric)
          fd:write(rlCurric .. "\n\n")
          fd:flush()
        end
    end

end -- end local function trainModel


--[[Auxiliary methodes]]
--TODO: do sthg about this piece of code repeated evrywhere need to refactor this!

function idx2key(file)
    local f = io.open(file, 'r')
    local t = {}
    local cnt = 0
    for line in f:lines() do
        local c = {}
        for w in line:gmatch '([^%s]+)' do
            table.insert(c, w)
        end
        t[tonumber(c[2])] = c[1]
        cnt = cnt +1
    end
    return t, cnt
end

function key2idx(file)
    local f = io.open(file, 'r')
    local t = {}
    local cnt = 0
    for line in f:lines() do
        local c = {}
        for w in line:gmatch '([^%s]+)' do
            table.insert(c, w)
        end
        t[c[1]] = tonumber(c[2])
        cnt = cnt +1
    end
    return t, cnt
end

function loadOnmtDictionary(file)
    local f = io.open(file, 'r')
    local wordVocab = onmt.utils.Dict.new()
    for line in f:lines() do
        local c = {}
        for w in line:gmatch '([^%s]+)' do
            table.insert(c, w)
        end
        wordVocab:add(c[1], tonumber(c[2]))
    end

    return {
    words = wordVocab,
    features = {}
  }
end

local function main()
  local requiredOptions = {
    "data_file",
    "valid_data_file",
    "save_model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  reseed()

  -- Create the data loader class.
  if not opt.json_log then
      if not opt.just_gen then
          print('Loading data from \n' .. opt.data_file .. '\n' .. opt.valid_data_file)
      end
  end

    -- Create the data loader class.
  print('loading data...')
  local trainData = onmt.data.WikiDataset2.new(opt, opt.data_file)
  local validData = onmt.data.WikiDataset2.new(opt, opt.valid_data_file)
  local testData
  if opt.test then
    testData = onmt.data.WikiDataset2.new(opt, opt.test_data_file)
  end

  if opt.maskNoneField then
    print("* Predictions with no DUMMY record")
    trainData:maskInputNoneField()
    validData:maskInputNoneField()
    if opt.test then testData:maskInputNoneField() end
  end

  print('loading dicts ...')
  idx2word_data, voc_data_size = idx2key(opt.data_dict)
  idx2word_text, voc_text_size = idx2key(opt.text_dict)
  print('done!')


  local dataset = {
      dicts = {
      src = loadOnmtDictionary(opt.data_dict),
      tgt = loadOnmtDictionary(opt.text_dict)
  }}
  assert(dataset.dicts.src.words:size()==voc_data_size)
  assert(dataset.dicts.tgt.words:size()==voc_text_size)

  g_tgtDict = dataset.dicts.tgt.words
  g_srcDict = idx2word_data
  g_targetDictEd = idx2word_text

  --TODO need to implement this if using reconstruction for our problem!!!!
  local tripV   -- vocabulary for each element in a triple (for rec)
  if opt.discrec then
      tripV = {dataset.dicts.src.rows:size(), dataset.dicts.src.cols:size(), dataset.dicts.src.cells:size()}
  end

  trainData:setBatchSize(opt.max_batch_size, opt.dataEncoder, opt.test)
  validData:setBatchSize(opt.max_batch_size, opt.dataEncoder, opt.test)
  if opt.test then testData:setBatchSize(opt.max_batch_size, opt.dataEncoder, opt.test) end

  if not opt.json_log then
      if not opt.just_gen then
        print(string.format(' * vocabulary size: source = %d; target = %d',
                            voc_data_size, voc_text_size))
        print(string.format(' * maximum sequence length: source = %d; target = %d',
                            trainData.max_dataseq_l, trainData.max_textseq_l))
        print(string.format(' * number of training instances: %d', trainData.length))
        print(string.format(' * maximum batch size: %d', opt.max_batch_size))
      end
  end

  if not opt.json_log then
      if not opt.just_gen then
          print('Building model...')
      end
  end

    if not opt.just_gen and not opt.evalAln then
        print(opt)
        if not opt.just_eval and not opt.scoresomethings then
            timeStamp = os.date():gsub(" ", "_"):gsub(":", "_")
            logFile = '../../oTrained/train_' .. timeStamp .. '.log'
            fd = io.open(logFile, 'w')
            fd:write(configDetails())
            fd:write("\n\n")
            fd:flush()
        end
    end

    -- opt for dtaAligner
    optDTA = {
        dataEncoder = opt.dataEncoder,
        vocabSize = dataset.dicts.src.words:size(),
        encDim = opt.enc_emb_size,
        decDim = opt.rnn_size,
        nLayers = opt.enc_layers,
        pool = opt.pool or "mean",
        effectiveDecLayers = opt.layers*2,
        dropout = opt.enc_dropout,
        relu = opt.enc_relu,
        wordVecSize = opt.word_vec_size,
        tanhOutput = opt.enc_tanh_output,
        input_feed = opt.input_feed,
        rnn_size = opt.rnn_size,
        preembed = opt.preembed,
        word_vec_size = opt.word_vec_size,
        preembed_datavoc = opt.preembed_datavoc,
        train = 1
    }

    local model = {}

    local verbose = true

    if opt.hdec then
        -- make sentence decoder first
        local sentDecoder =  onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose, tripV)
        model.decoder = onmt.Models.buildHirarchicalDecoder(opt, verbose, sentDecoder)
    else
        -- make decoder first
        opt.max_textseq_l = trainData.max_textseq_l
        model.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose, tripV)
    end

    -- send to gpu immediately to make cloning things simpler
    onmt.utils.Cuda.convert(model.decoder)

    model.encoder = onmt.WikiDataEncoder(optDTA)

    onmt.utils.Cuda.convert(model.encoder)

    --TODO: this is sharing of embedding matrices between encoder/decoder for the moment our vocabularies are different. See then if we can use a single matrix too.
    -- share all the things
    --assert(model.encoder.lut.weight:size(1) == model.decoder.inputNet.net.weight:size(1))
    --model.encoder.lut:share(model.decoder.inputNet.net, 'weight', 'gradWeight')
    model.encoder:shareTranforms()

    trainModel(model, trainData, validData, testData, dataset, nil)

    if fd then
        fd:close()
    end

    print("done.")
end

main()
