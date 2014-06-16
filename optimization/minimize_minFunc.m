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