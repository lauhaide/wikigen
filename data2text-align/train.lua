--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 16/02/17
-- Time: 11:44
-- To change this template use File | Settings | File Templates.
--

require 'nn'
require 'rnn'
require 'hdf5'
require 'nngraph'
require 'optim'
require 'image'
require 'lfs'

require 'dta.data'
require 'dta.ldepData'
require 'evaluationUtils'
require 'utils.fileUtils'
require 'dta.milTriplesSM2Deep'
require 'dta.corruptGrammaticality'


cmd = torch.CmdLine()

cmd:text("")
cmd:text("**Running options**")
cmd:text("")
cmd:option('-doeval', false, [[Only run evaluation and/or test test modes]])
cmd:option('-verbose', false, [[Print further system messages]])
cmd:option('-dobaseline', false,
    [[Only run evaluation and/or test test modes with the word embedding baseline scorer]])
cmd:option('-multitask', false, [[Run multitask on sentence encoding]])

cmd:text("")
cmd:text("**Evaluation options**")
cmd:text("")
cmd:option('-genNegValid', false, [[Regenerate the set of distractor sentences for evaluation]])
cmd:option('-distractorSet', '', [[File name (with path) file to load from or save to generated
                                    distractor set for ranking evaluation]])
cmd:option('-waEvaluationSet', '',
    [[File name (with path) of text file containing selected item ids with manual
    gold annotated alignments for word alignment based evaluation]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file', '', [[Path to the training *.hdf5 file from Preprocess.py]])
cmd:option('-valid_data_file', '', [[Path to validation *.hdf5 file from Preprocess.py]])
cmd:option('-test_data_file', '', [[Path to test *.hdf5 file from Preprocess.py]])
cmd:option('-data_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-text_dict', '', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-preembed_datavoc', '', [[Use pre-trained vectors for data vocabulary dict.]])
cmd:option('-preembed_textvoc', '', [[Use pre-trained vectors for text vocabulary dict.]])
cmd:option('-weights_file', '', [[Path to model weights (*.t7 file)]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-linDepData', '../../trained/input/tmp-lingDepsTrain_II.hdf5', [[Dataset for multitask training.]])


cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-rnn_size', 200, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 200, [[Word embedding sizes]])
cmd:option('-dataEncoder', 0, [[Type of encoding for triple/property data.]])
cmd:option('-textEncoder', 0, [[Type of encoding for sentences/text data.]])
cmd:option('-selfAttn', false, [[Use self attention for the sentence encoder.]])
cmd:option('-preembed', false, [[Initialise embedding layers with pre-trained vectors]])
cmd:option('-propNameKB', false, [[Whether to represent the property name as a short sequence of words or
                                    a unique KB symbol with separate vocabulary matrix.]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-epochs', 7, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd (vanilla SGD), adagrad, adadelta, adam]])
cmd:option('-learning_rate', 0.01, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings: sgd =1,
                                adagrad = 0.1, adadelta = 1, adam = 0.1]])
cmd:option('-max_batch_l', 100, [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])
cmd:option('-min_batch_l', 100, [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])
cmd:option('-lr_decay', 1e-4, [[Decay learning rate]])
cmd:option('-use_cuda', false, [[Use GPU]])



function configDetails()
    --return string.format("epochs: %d\nbatch size: %d\nemb dim: %d\nhidden dim: %d\noptim algo: ".. opt.optim .."\nlearning rate: %.4f\nmodel:%d \nuse pre-trained vectors initialisation: " .. tostring(opt.preembed),
    --    opt.epochs, opt.max_batch_l, opt.word_vec_size, opt.rnn_size, opt.learning_rate, opt.model)

  local options = ""
  for k, v in pairs(opt) do
      options = options .. "-" .. k .." " .. tostring(v) .. " "
  end
  return options
end


function dataDetails(train_data)
   return string.format("\nnb training examples: %d, \nnb negative training examples: %d\nData vocab size: %d, Text vocab size: %d \n\n",
       train_data.length, train_data.length, train_data.datavoc_size, train_data.textvoc_size)
end


function train(scorer, criterion, train_data, lingDepDataset, auxiliarModel, mtCriterion)
  local fd = io.open(logFile, 'w')
  fd:write(configDetails())
  fd:write(dataDetails(train_data))
  fd:flush()
  print("Logging to " .. logFile)

  local optim_func, optim_state = get_training_method_details()
  local mtOptimFunc, mtOptimState =  get_training_method_Multitask()
  local minibatches = train_data:cacheNSMiniBatches()

  local batch, batch_neg, batch_loss, total_loss, mtTotalLoss, mtLoss, mtBatches
  timer = torch.Timer()
  ProFi = require 'ProFi'
  ProFi:start()
  for i=1,opt.epochs do
    mtBatches = 1
  --for i=1,1 do
    total_loss = 0
    mtTotalLoss = 0
    print(string.format("Running epoch: %d",i))
    for i=1, train_data:cachedMBatchesCount() do
    --for i=1, 500 do
        print(string.format("minibatch: %d", i))
        batch, batch_neg = unpack(minibatches[i])

        if opt.multitask and (torch.rand(1)[1]) > 0.5 and mtBatches < lingDepDataset:cachedMBatchesCount() then --last mbatch is empty? --*0.6) > 0.5
            print(string.format("multitask minibatch: %d", mtBatches))

            --multitask train
            mtLoss = multitaskTrainBatch(auxiliarModel, mtCriterion, lingDepDataset:getMultitasMiniBatches()[mtBatches], mtOptimFunc, mtOptimState)
            mtTotalLoss = mtTotalLoss + mtLoss
            mtBatches = mtBatches + 1
        end

        batch_loss = train_batch(scorer, criterion, batch, batch_neg, optim_func, optim_state)
        --print("Batch loss", batch_loss)
        total_loss =  total_loss + batch_loss
        --os.exit()
        if i % 50 == 0 then
            collectgarbage()
            collectgarbage()
        end
    end
    ProFi:stop()
    ProFi:writeReport( profFile )

    print("Train", total_loss)
    fd:write(string.format("Train loss at epoch %d:  %.4f \n", i, total_loss))
    fd:flush()
        --eval???
    if opt.multitask then
        print("Train multitask ", mtTotalLoss)
    end

    save_parameters(scorer, weightsFile:gsub(".t7", string.format("-%d.t7",i)):gsub("trained","trained/tmp"))
    print(string.format("save weights saved to file " .. weightsFile .. " at epoch %d",i))
  end

  timer:stop()
  fd:write('\n' .. timer:time().real .. ' seconds')
  fd:close()
end

--this is to train the model with sentence grammatical detection auxiliary task, finally not used
function get_training_method_Multitask()
  local optim_state, optim_func
    optim_state = {
           learningRate = opt.learning_rate
    }
    optim_func = optim.sgd
  return optim_func, optim_state
end

function get_training_method_details()
  local optim_state, optim_func
  if opt.optim ==  'sgd' then
    optim_state = {
           learningRate = opt.learning_rate
    }
    optim_func = optim.sgd
  elseif  opt.optim == 'adagrad' then
    optim_state = {
        learningRate = opt.learning_rate,
    }
    optim_func = optim.adagrad
  elseif  opt.optim == 'adam' then
    optim_state = {
        learningRate = opt.learning_rate,
    }
    optim_func = optim.adam
  end
  return optim_func, optim_state
end

--[[NOT USED IN OUR EXPERIMENTS]]
--[[gets random negative sentences from an out of biography Wikipedia dataset]]
--randomly picks up sentences
function getNegativeSamplesOOBio(setLength, nbNegComp, dataneg)  --[[Get negative training pairs for evaluation]]
  if opt.genNegValid then
    --[[Generate new pairs]]
    print("Generating new pairs of negative examples for the evaluation")
    local perm_neg
    local distrIDXs =  torch.Tensor(setLength, nbNegComp)
    for i =1, setLength do
        perm_neg = torch.randperm(dataneg.length)
        distrIDXs[i] = perm_neg:sub(1, nbNegComp)
    end
    torch.save(opt.distractorSet, distrIDXs)
    return distrIDXs
  else
    --[[Load existing negative pairs]]
    return torch.load(opt.distractorSet)
  end
end

function eval(fragsorer, evaluationset, idx2word_data, idx2word_text, resultDir, yawatEvalFile, rankingMiniBatch)

  local totalComp = rankingMiniBatch
  local nbNegComp = totalComp -1
  local drawx = 0
  local sumRank = 0
  local distrIDXs
  local setLength = evaluationset.length

  --get distractors for ranking based evaluation
  if rankingMiniBatch > 1 then
      distrIDXs = evaluationset:getNegativeSamplesInBioBuckets(opt, nbNegComp)
  end

  --get selected examples for word alignment -based evaluation
  local yawatEvalSet = readYawatEvalSelection(yawatEvalFile)
  print("read item selection for wa-based evaluation...")

  local minibatches = evaluationset:getEvaluationMiniBatches(rankingMiniBatch, distrIDXs)
  local waMatrices = {}
  local mbatch, scores, frag_score, dbp, text, curids, alignment, dataseq, seqFragments
  for i=1, evaluationset:cachedEvaluationMBatchesCount() do
  --for i=1, 1000 do
    print(string.format("minibatch: %d", i))
    mbatch = unpack(minibatches[i])
    dbp, text, curids = unpack(mbatch)

    enc_data1, enc_text1, scores, frag_score = unpack(fragsorer:forward({dbp, text}))

    --[[drawing first 10 sets of corrext and distractor heatmaps alignments]]
    local datasing

    for j=1, scores:size()[1], rankingMiniBatch do
        local y, idx = torch.sort(scores:sub(j,j+ rankingMiniBatch -1), true)
        local rank = torch.nonzero(idx:eq(1))[1][1]
        print(string.format("Correct ranked at position %d /%d",rank, totalComp))
        sumRank = sumRank + rank

        --[[writing alignmet for waq measure]]
        if yawatEvalSet:eq(curids[j][1]):sum() > 0 then
            print("write word alignments curid="..tostring(curids[j][1]) .. " sentence ID=" .. tostring(curids[j][2]-1) )
            datasing = evaluationset:getSingleDataSequence(dbp, j) --returns right aligned sequence
            alignment, dataseq = adjustMatrix2Sequence(
                {datasing, text[j]}, evaluationset.max_prop_l, evaluationset.max_dataseq_l, frag_score[j], idx2word_text, LPADDING)
            seqFragments = sequenceFragScores({datasing, text[j]}, frag_score[j], LPADDING)
            table.insert(waMatrices, {curids[j], alignment, dataseq, seqFragments, scores[j], text[j][text[j]:gt(0)]})
        end
    end
    drawx = drawx + 1
  end
  ---TODO last parameters should depend on whether we do baseline or not!!!! if baseline then false
  evalAlignment(waMatrices, idx2word_text, resultDir, true)

  print(string.format("Average rank is %.2f", sumRank/setLength))
  print("results in "..resultDir)
end

function main()
  -- parse input params
  opt = cmd:parse(arg)
  require 'utils.cuda'

  -- Create the data loader class.
  print('loading data...')
  local train_data = data.new(opt, opt.data_file)
  local valid_data = data.new(opt, opt.valid_data_file)
  --local test_data = data.new(opt, opt.test_data_file)
  local lingDepDataset
  if opt.multitask then
    lingDepDataset = ldepData.new(opt, opt.linDepData)
  end
  print('done!')

  local idx2word_data, voc_data_size = idx2key(opt.data_dict)
  local idx2word_text, voc_text_size = idx2key(opt.text_dict)

  print(dataDetails(train_data))

  if opt.verbose then
      print("print example to verify indexing encoding...")
      local b = train_data:getStructBucket(1)
      print(b)
      local dbp, text, bucket_size, data_seq_length, prop_seq_length, prop_val_seq_length, text_seq_length = unpack(b)
      local dprop, dval = unpack(dbp)
      print(dbp)
      print(text)
      --print 1 pair
      print(string.format("training example %d ",1))
      for i=1, data_seq_length do
        print(seqToText(idx2word_data, dprop[1][i]) .. " | " .. seqToText(idx2word_data, dval[1][i]))
      end
      print(seqToText(idx2word_text, text[1][1]))
      b = train_data:getSingleSequenceBucket(1)
      local dbp, text, bucket_size, data_seq_length, prop_seq_length, text_seq_length = unpack(b)
      for i=1, data_seq_length do
          print(seqToText(idx2word_data, dbp[1][1][i]))
      end

  end

  if opt.doeval == false and opt.dobaseline==false then
    profFile = '../../trained/ProfilingReport_' .. os.date():gsub(" ", "_"):gsub(":", "_") .. '.txt'
    logFile = '../../trained/train_' .. os.date():gsub(" ", "_"):gsub(":", "_") .. '.log'
    weightsFile = string.format("../../trained/m_%d_%d", opt.dataEncoder, opt.textEncoder) .. "scorer_weights_" .. os.date():gsub(" ", "_"):gsub(":", "_")  .. ".t7"

    local scorer, sentEncoder = train_alignment_scorer(opt, voc_data_size, voc_text_size, train_data)
    local criterion = getCriterion(opt)
    local mtCriterion = multitaskCriterion()

    local auxiliarModel
    if opt.multitask then
        --embedding layer and biLSTM ebcoder are shared with this model
        auxiliarModel = gramticalJudgementModel(opt, sentEncoder)
    end

    if opt.verbose then
       graph.dot(scorer.fg, 'Forward Graph','scorer5_fg')
       --graph.dot(scorer.bg, 'Backward Graph','scorer5_bg')
    end

    if opt.train_from:len() ~= 0 then
        print("Setting model parameters to start from ..." .. opt.train_from)
        load_parameters(scorer, opt.train_from)
    end

    train(scorer, criterion, train_data, lingDepDataset, auxiliarModel, mtCriterion)

    --save model parameters
    save_parameters(scorer, weightsFile)
    print("finish training, weights saved to file " .. weightsFile)

  else
    local outDir, fragsorer
    if opt.dobaseline then
      require 'dta.WEBaseline'
      print("get WE baseline model")
      outDir = "../../trained/results/WEBaseline/"
      fragsorer = embBaselineScorer(opt, voc_data_size, voc_text_size)
    else
      weightsFile = opt.weights_file
      outDir = "../../trained/results/" .. getWeightsName(opt.weights_file) .. "/"
      fragsorer = load_alignment_scorer(opt, voc_data_size, voc_text_size, train_data, weightsFile)
    end

    --print(fragsorer.forwardnodes[5].data.module.modules[1].weight[1]) --this is lookuptable in data encoding part **debugging**
    if isDir(outDir) or lfs.mkdir(outDir) then
        print("start evaluation...")
        local rankingMiniBatch = 15
        eval(fragsorer, valid_data, idx2word_data, idx2word_text, outDir, opt.waEvaluationSet, rankingMiniBatch)
    else
        print("Could not create output folder: " .. outDir)
    end
  end


end


main()
