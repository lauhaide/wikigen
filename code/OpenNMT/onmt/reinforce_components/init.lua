local reinforce_components = {}

reinforce_components.SimilarityScorer = require('onmt.reinforce_components.SimilarityScorer')
reinforce_components.WAScorerB1 = require('onmt.reinforce_components.WAScorerB1')
reinforce_components.WAScorerCombined = require('onmt.reinforce_components.WAScorerCombined')
reinforce_components.WAScorerCombinedFpLrG = require('onmt.reinforce_components.WAScorerCombinedFpLrG')
reinforce_components.WAScorerCombined2 = require('onmt.reinforce_components.WAScorerCombined2')
reinforce_components.WAScorerCombinedFpLrGSelective = require('onmt.reinforce_components.WAScorerCombinedFpLrGSelective')

return reinforce_components
