% LEARN_GP_HYPERPARAMETERS actively learn GP hyperparameters.
%
% This is an implementation of the method for actively learning GP
% hyperparameters described in
%
%   Garnett, R., Osborne, M., and Hennig, P. Active Learning of Linear
%   Embeddings for Gaussian Processes. (2014). 30th Conference on
%   Uncertainty in Artificial Intelligence (UAI 2014).
%
% Given a GP model on a function f:
%
%   p(f | \theta) = GP(f; mu(x; \theta), K(x, x'; \theta)),
%
% this routine sequentially chooses a sequence of locations X = {x_i}
% to make observations with the goal of learning the GP
% hyperparameters \theta as quickly as possible. This is done by
% maintaining a probabilistic belief p(\theta | D) and selecting
% each observation location by maximizing the Bayesian active
% learning by disagreement (BALD) criterion described in
%
%   N. Houlsby, F. Huszar, Z. Ghahramani, and M. Lengyel. Bayesian
%   Active Learning for Classification and Preference
%   Learning. (2011). arXiv preprint arXiv:1112.5745 [stat.ML].
%
% This implementation uses the approximation to BALD described in the
% Garnett, et al. paper above, which relies on the "marginal GP" (MGP)
% method for approximate GP hyperparameter marginalization.
%
% See demo/demo.m for a simple example usage.
%
% Dependencies
% ------------
%
% This code is written to be interoperable with the GPML MATLAB
% toolbox, available here:
%
%   http://www.gaussianprocess.org/gpml/code/matlab/doc/
%
% The GPML toolbox must be in your MATLAB path for this function to
% work. This function also depends on the gpml_extensions repository,
% available here:
%
%   https://github.com/rmgarnett/gpml_extensions/
%
% as well as the marginal GP (MGP) implementation available here:
%
%   https://github.com/rmgarnett/mgp/
%
% Both must be in your MATLAB path. Finally, the optimization of
% the GP log posterior requires Mark Schmidt's minFunc function:
%
%   http://www.di.ens.fr/~mschmidt/Software/minFunc.html
%
% Usage
% -----
%
%   results = learn_gp_hyperparameters(problem, model, varargin)
%
% Required inputs:
%
%   problem: a struct describing the active learning problem,
%            containing fields:
%
%      num_evaluations: the number of observations to select
%     candidate_x_star: the set of observation locations available
%                       for selection
%                    f: a function handle that returns the function
%                       observation at a given point, which will be
%                       called as
%
%                         y_star = problem.f(x_star)
%
%     model: a struct describing the GP model, containing fields:
%
%          inference_method: a GPML inference method
%                            (optional, default: @exact_inference)
%             mean_function: a GPML mean function
%                            (optional, default: @zero_mean)
%       covariance_function: a GPML covariance function
%                likelihood: a GPML likelihood
%                            (optional, default: @likGauss)
%                     prior: a function handle to a hyperparameter
%                            prior p(\theta) (see priors.m in
%                            gpml_extensions)
%
% Optional inputs (specified as name/value pairs):
%
%   'minFunc_options': a struct containing options to pass to
%                      minFunc when optimizing the log posterior,
%                      default:
%
%                        .Display     = 'off'
%                        .MaxFunEvals = 300
%
%      'num_restarts': the number of random restarts to use when
%                      optimizing the log posterior, default: 1
%
% Outputs:
%
%   results: a struct describing the active learning process,
%            containing fields:
%
%                .chosen_x: the chosen observation locations
%                           (problem.num_evalutations x D)
%                .chosen_y: the associated observation values
%                           (problem.num_evalutations x 1)
%     .map_hyperparameters: an array of GPML hyperparameter structs
%                           containing the MAP hyperparameters
%                           learned after every evaluation
%          .map_posteriors: an array of GPML posterior structs
%                           corresponding to
%
%                             (D_i, \hat{\theta}_i),
%
%                            where D_i is the first i observations,
%
%                             D_i = {(x_k, y_k)} (k = 1..i),
%
%                           and \hat{\theta}_i is the MAP
%                           hyperparameters learned from D_i
%                           (contained in results.map_hyperparameters)
%
% See also GP, MGP, MINFUNC.

% Copyright (c) 2014 Roman Garnett.

function results = learn_gp_hyperparameters(problem, model, varargin)

  % parse optional inputs
  parser = inputParser;

  addParamValue(parser, 'num_restarts', 1);
  addParamValue(parser, 'minFunc_options', ...
                        struct('Display',     'off', ...
                               'MaxFunEvals', 300));

  parse(parser, varargin{:});
  options = parser.Results;

  % check model specification
  if (~isfield(model, 'inference_method') || isempty(model.inference_method))
    model.inference_method = @exact_inference;
  end

  if (~isfield(model, 'mean_function') || isempty(model.mean_function))
    model.mean_function = {@zero_mean};
  end

  if (~isfield(model, 'likelihood') || isempty(model.likelihood))
    model.likelihood = @likGauss;
  end

  % add prior to inference method if not already incorporated
  if (~isempty(strfind(func2str(model.inference_method), 'inference_with_prior')))
    model.inference_method = ...
        add_prior_to_inference_method(model.inference_method, model.prior);
  end

  % choose first point closest to mean of available points
  residuals = bsxfun(@minus, problem.candidate_x_star, ...
                             mean(problem.candidate_x_star));
  [~, ind] = min(sum(residuals.^2, 2));

  x = problem.candidate_x_star(ind, :);
  y = problem.f(x);

  problem.candidate_x_star = ...
      problem.candidate_x_star([1:(ind - 1), (ind + 1):end], :);

  % allocate output
  results.map_embeddings = repmat(model.prior(), [problem.num_evaluations, 1]);

  for i = 1:problem.num_evaluations

    % always start optimization at previous MAP hyperparameters
    if (i > 1)
      initial_hyperparameters = results.map_hyperparameters(i - 1);
    else
      initial_hyperparameters = [];
    end

    % find MAP hyperparameters
    results.map_hyperparameters(i) = ...
        minimize_minFunc(model, x, y, ...
                         'initial_hyperparameters', initial_hyperparameters, ...
                         'num_restarts',            options.num_restarts, ...
                         'minFunc_options',         options.minFunc_options);

    [~, ~, results.map_posteriors(i)] = gp(results.map_hyperparameters(i), ...
            model.inference_method, model.mean_function, ...
            model.covariance_function, model.likelihood, x, y);

    % allocate output
    if (i == 1)
      results.map_posteriors = ...
          repmat(results.map_posteriors(1), [problem.num_evaluations, 1]);
    end

    if (i == problem.num_evaluations)
      break;
    end

    ind = select_next_point(results.map_hyperparameters(i), ...
                            model, x, results.map_posteriors(i), ...
                            problem.candidate_x_star);

    x_star = problem.candidate_x_star(ind, :);

    x = [x; x_star];
    y = [y; problem.f(x_star)];

    problem.candidate_x_star = ...
        problem.candidate_x_star([1:(ind - 1), (ind + 1):end], :);
  end

  results.chosen_x = x;
  results.chosen_y = y;

end