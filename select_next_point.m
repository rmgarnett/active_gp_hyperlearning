% SELECT_NEXT_POINT selects next point for GP hyperparameter learning.
%
% This function determines which point in a given set of candidates
% is the most informative for learning GP hyperparameter
% learning. We do this by maximizing the Bayesian active learning
% by disagreement (BALD) objective:
%
%   H[f | x, D] - E_\theta[f | x , D, \theta]
%
% We use the marginal GP approximation (MGP) to approximate this
% objective.
%
% Usage
% -----
%
%   ind = select_next_point(map_hyperparameters, model, x, y, x_star)
%
% Inputs:
%
%   map_hyperparameters: a GPML hyperparameter struct containing
%                        the MAP hyperparameters given (X, y)
%                 model: a struct describing the GP model,
%                        containing fields:
%
%        inference_method: a GPML inference method
%           mean_function: a GPML mean function
%     covariance_function: a GPML covariance function
%              likelihood: a GPML likelihood
%
%                     x: the observation locations (N x D)
%                     y: the corresponding observation values
%                        (N x 1)
%                x_star: the candidate observation locations (N_* x D)
%
% Outputs:
%
%   ind: an index into x_star identifying the observation location
%        to select next
%
% See also LEARN_GP_HYPERPARAMETERS.

% Copyright (c) 2014 Roman Garnett.

function ind = select_next_point(map_hyperparameters, model, x, y, x_star)

  [~, ~, ~, f_star_variance_mgp, ...
   ~, ~, ~, f_star_variance_gp  ] = ...
      mgp(map_hyperparameters, model.inference_method, model.mean_function, ...
          model.covariance_function, model.likelihood, x, y, x_star);

  scores = f_star_variance_mgp ./ f_star_variance_gp;

  [~, ind] = max(scores);

end