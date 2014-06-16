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

    results.map_hyperparameters(i) = ...
        minimize_minFunc(initial_hyperparameters, model, x, y, ...
                         'num_restarts',    options.num_restarts, ...
                         'minFunc_options', options.minFunc_options);

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