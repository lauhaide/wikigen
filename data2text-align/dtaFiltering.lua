--
-- Created by IntelliJ IDEA.
-- User: lperez
-- Date: 31/07/17
-- Time: 11:30
-- To change this template use File | Settings | File Templates.
--

package.path = package.path .. ';dta/?.lua'


require 'nn'
require 'rnn'
require 'hdf5'
require 'nngraph'

require 'dta.data'
require 'dta.milTriplesSM2Deep'
require 'evaluationUtils'
require 'utils.fileUtils'

cmd = torch.CmdLine()

cmd:text("")
cmd:text("**Running options**")
cmd:text("")
cmd:option('-max_batch_l', 100, [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])
cmd:option('-min_batch_l', 100, [[If blank, then it will infer the max batch size from validation
                               data. You should only use this if your validation set uses a different
                               batch size in the preprocessing step]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-weights_file', '', [[Path to model weights (*.t7 file)]])
cmd:option('-data_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-text_dict', '', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-data_file','', [[Path to the training *.hdf5 file from Preprocess.py]])
cmd:option('-wsthreshold', 0.0, [[Threshold to determine, given an alignment score, if there exist alignment or not.]])
cmd:option('-proportion', 0.0, [[This is used to create a filtered dataset (i.e. filtering sentences that have low
                                alignment). Proportion of aligned words in the sentence, for the sentence to be
                                selected the proportion of aligned words should be higher than *proportion* parameter
                                value.]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")
cmd:option('-rnn_size', 200, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 200, [[Word embedding sizes]])
cmd:option('-dataEncoder', 0, [[Type of encoding for triple/property data.]])
cmd:option('-textEncoder', 0, [[Type of encoding for sentences/text data.]])
cmd:option('-selfAttn', false, [[Use self attention for the sentence encoder.]])


cmd:option('-use_cuda', true, [[Use GPU]])

--[[Extracts alignment information from DTA-aligner model to be used by encoder-decoder model variants.
-- Filtering and Guided.
-- ]]
function alignmentFilter(fragsorer, dataset, datasetName, outDir, idx2word_text, idx2word_data)
    local rankingMiniBatch = 1 --get batches with no ranking preparation
    local minibatches = dataset:getEvaluationMiniBatches(rankingMiniBatch, nil)
    local outCurid = assert(io.open(outDir .. datasetName .. "-fcurid.possib_0.6.txt", "w"))
    local alnLabels = assert(io.open(outDir .. datasetName .. "-alnLabels.possib_0.6.txt", "w"))

    local sents = {}
    local passSents = {}
    local accWords = {}
    local alnAccWords = {}
    for ii=1, MAX_SENT_NB do
        table.insert(sents, 0)
        table.insert(passSents, 0)
        table.insert(accWords, 0)
        table.insert(alnAccWords, 0)
    end

    totalSents, totalAligns = 0, 0
    for i=1, dataset:cachedEvaluationMBatchesCount() do
        print(string.format("minibatch: %d", i))
        mbatch = unpack(minibatches[i])
        dbp, text, curids = unpack(mbatch)
        totalSents = totalSents + curids:size()[1]

        enc_data1, enc_text1, scores, frag_score = unpack(fragsorer:forward({dbp, text}))

        for j=1, scores:size()[1] do
            datasing = dataset:getSingleDataSequence(dbp, j) --returns right aligned sequence
            seqFragments = sequenceFragScores({datasing, text[j]}, frag_score[j], LPADDING)

            --if debugging:
            --print(multiseqToText(idx2word_data, datasing))
            --print(seqToText(idx2word_text, text[j]))

            sents[curids[j][2]] = sents[curids[j][2]] + 1 --increment the nb of ith sentences

            -- X% of alignment per sentence for filtering
            local nbWordAlns, totalWords = aligns(seqFragments, dataset.NoneField, opt.wsthreshold, datasing)

            if (nbWordAlns / totalWords) > opt.proportion then
                --print(curids[j][1] .. "." .. curids[j][2]-1) --sentence list in python and yawat start from 0 index
                outCurid:write(curids[j][1] .. "." .. curids[j][2]-1 .. "\n")
                totalAligns = totalAligns +1
                passSents[curids[j][2]] = passSents[curids[j][2]] + 1 --increment the nb of ith sentences that passed the filter
            end

            accWords[curids[j][2]] = accWords[curids[j][2]] + totalWords
            alnAccWords[curids[j][2]] = alnAccWords[curids[j][2]] + nbWordAlns

            -- word based target alignment labels
            alnLabels:write(
                curids[j][1] .. "." .. curids[j][2]-1 .. "\t" ..
                alignmentLabels(seqFragments, dataset.NoneField, opt.wsthreshold, datasing) .. "\n")


        end

    end
    outCurid:close()
    alnLabels:close()
    print(string.format("%d aligned sentences out of %d ", totalAligns, totalSents))

    print("* ith sentence proportion of aligned sentences")
    for ii=1, MAX_SENT_NB do
        if sents[ii] ~= 0 then
        print(string.format("%d : %.2f (%d/%d) ", ii, passSents[ii]/sents[ii]*100, passSents[ii], sents[ii]))
        end
    end

    print("* ith sentence average of aligned words")
    for ii=1, MAX_SENT_NB do
        if sents[ii] ~= 0 then
        print(string.format("%d : %.2f ", ii, alnAccWords[ii]/sents[ii]))
        end
    end

    print("* ith sentence average word length")
    for ii=1, MAX_SENT_NB do
        if sents[ii] ~= 0 then
        print(string.format("%d : %.2f ", ii, accWords[ii]/sents[ii]))
        end
    end

end

function main()
    -- parse input params
    opt = cmd:parse(arg)
    require 'utils.cuda'

    print("hello")
    -- load datasets to filter
    print('loading data...')
    local idx2word_data, voc_data_size = idx2key(opt.data_dict)
    local idx2word_text, voc_text_size = idx2key(opt.text_dict)

    -- load DTA aligner
    local outDir, fragsorer
    local weightsFile = opt.weights_file
    outDir = "../../trained/results/" .. getWeightsName(opt.weights_file) .. "/filtered/"
    fragsorer = load_alignment_scorer(opt, voc_data_size, voc_text_size, dataset, weightsFile)

    --Max. number of sentences per document
    --read this from the dataset, need to be passed with pre-proc data
    MAX_SENT_NB = 20

    -- run aligner filter and save new datasets
    if isDir(outDir) or lfs.mkdir(outDir) then

        local index = string.find(opt.data_file, "/[^/]*$")
        local outFileName = string.sub(opt.data_file, index+1)
        index = string.find(outFileName, ".[^.]*$")
        outFileName = string.sub(outFileName, 1, index-1)

        local dataset = data.new(opt, opt.data_file)


        alignmentFilter(fragsorer, dataset, outFileName, outDir, opt.threshold, idx2word_text, idx2word_data)
    end

    print("done.")

end

main()
