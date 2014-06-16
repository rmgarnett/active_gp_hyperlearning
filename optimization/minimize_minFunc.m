% MINIMIZE_MINFUNC optimize GP hyperparameters with random restart.
%
% This implements GP hyperparameter optimization with random
% restart. Each optimization is accomplished using Mark Schmidt's
% minFunc function:
%
%   http://www.di.ens.fr/~mschmidt/Software/minFunc.html
%
% For each restart, a new set of hyperparameters will be drawn from a
% specified hyperparameter prior p(\theta). The user may optionally
% specify the initial hyperparameters to use for the first
% optimization attempt.
%
% Usage
% -----
%
%   [best_hyperparameters, best_nlZ] = minimize_minFunc(model, x, y, varargin)
%
% Required inputs:
%
%   model: a struct describing the GP model, containing fields:
%
%        inference_method: a GPML inference method
%           mean_function: a GPML mean function
%     covariance_function: a GPML covariance function
%              likelihood: a GPML likelihood
%                   prior: a function handle to a hyperparameter prior
%                          p(\theta) (see priors.m in gpml_extensions)
%
%       x: the observation locations (N x D)
%       y: the observation values (N x 1)
%
% Optional inputs (specified as name/value pairs):
%
%   'initial_hyperparameters': a GPML hyperparameter struct specifying
%                              the intial hyperparameters for the
%                              first optimization attempt (if not
%                              specified, will be drawn from the prior)
%
%           'minFunc_options': a struct containing options to pass to
%                              minFunc when optimizing the log
%                              posterior, default:
%
%                               .Display     = 'off'
%                               .MaxFunEvals = 300
%
%             'num_restarts': the number of random restarts to use
%                             when optimizing the log posterior,
%                             default: 1
%
%                             Note: this specifies the number of
%                             _re_starts; at least one optimization
%                             call will always be made.
%
% See also MINFUNC.

% Copyright (c) 2014 Roman Garnett

function [best_hyperparameters, best_nlZ] = minimize_minFunc(model, ...
          x, y, varargin)

  % parse optional inputs
  parser = inputParser;

  addParamValue(parser, 'initial_hyperparameters', []);
  addParamValue(parser, 'num_restarts', 1);
  addParamValue(parser, 'minFunc_options', ...
                        struct('Display',     'off', ...
                               'MaxFunEvals', 300));

  parse(parser, varargin{:});
  options = parser.Results;

  if (isempty(options.initial_hyperparameters))
    initial_hyperparameters = model.prior();
  else
    initial_hyperparameters = options.initial_hyperparameters;
  end

  f = @(hyperparameter_values) gp_optimizer_wrapper(hyperparameter_values, ...
          initial_hyperparameters, model.inference_method, ...
          model.mean_function, model.covariance_function, model.likelihood, ...
          x, y);

  [best_hyperparameter_values, best_nlZ] = ...
      minFunc(f, unwrap(initial_hyperparameters), options.minFunc_options);

  for i = 1:options.num_restarts
    hyperparameters = model.prior();

    [hyperparameter_values, nlZ] = ...
        minFunc(f, unwrap(hyperparameters), options.minFunc_options);

    if (nlZ < best_nlZ)
      best_nlZ = nlZ;
      best_hyperparameter_values = hyperparameter_values;
    end
  end

  best_hyperparameters = rewrap(initial_hyperparameters, ...
                                best_hyperparameter_values);

end