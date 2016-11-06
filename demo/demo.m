rng('default');

% Setup prediction GP. We'll use a constant mean and a squared
% exponential covariance. We must use mean/covariance functions that
% support an extended GPML syntax allowing the calculation of second
% partial dervivatives with respect to hyperparameters. The
% gpml_extensions package contains implementations of some common
% choices.

model.mean_function       = {@constant_mean};
model.covariance_function = {@isotropic_sqdexp_covariance};
model.likelihood          = @likGauss;

% initial hyperparameters
offset       = 1;
length_scale = 1.25;
output_scale = 2;
noise_std    = 0.75;

true_hyperparameters.mean = offset;
true_hyperparameters.cov  = log([length_scale; output_scale]);
true_hyperparameters.lik  = log(noise_std);

% Setup hyperparameter prior. We'll use independent normal priors on
% each hyperparameter.

% N(0, 0.5^2) priors on each log covariance parameter
priors.cov  = ...
    {get_prior(@gaussian_prior, 0, 1), ...
     get_prior(@gaussian_prior, 0, 1)};

% N(0.1, 0.5^2) prior on log noise
priors.lik  = {get_prior(@gaussian_prior, 0, 1)};

% N(0, 1) prior on constant mean
priors.mean = {get_prior(@gaussian_prior, 0, 1)};

model.prior = get_prior(@independent_prior, priors);
model.inference_method = ...
    add_prior_to_inference_method(@exact_inference, model.prior);

% generate demo data

num_points = 500;

x_star = linspace(-5, 5, num_points)';

mu = feval(model.mean_function{:},       true_hyperparameters.mean, x_star);
K  = feval(model.covariance_function{:}, true_hyperparameters.cov,  x_star);

K = (K + K') / 2;

y_star = mvnrnd(mu, K)';
y_star = y_star + exp(true_hyperparameters.lik) * randn(size(y_star));

% setup problem struct
problem.num_evaluations  = 15;
problem.candidate_x_star = x_star;

% function is a simple lookup table
problem.f                = ...
    @(x) (y_star(find(all(bsxfun(@eq, x, x_star), 2))));

% actively learn GP hyperparameters
results = learn_gp_hyperparameters(problem, model);

[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    gp(results.map_hyperparameters(end), model.inference_method, ...
       model.mean_function, model.covariance_function, model.likelihood, ...
       results.chosen_x, results.chosen_y, x_star, y_star);

report = sprintf('active: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities));
fprintf('%s\n', report);

x = results.chosen_x;
y = results.chosen_y;

figure(1);
set(gcf, 'color', 'white');
subplot(2, 1, 1);
plot_predictions;
title(report);

% compare to random sampling
ind = randperm(num_points, problem.num_evaluations);

x = x_star(ind, :);
y = y_star(ind);

map_hyperparameters_random = minimize_minFunc(model, x, y);

[~, ~, f_star_mean, f_star_variance, log_probabilities] = ...
    gp(map_hyperparameters_random, model.inference_method, ...
       model.mean_function, model.covariance_function, model.likelihood, ...
       x, y, x_star, y_star);

report = sprintf('random: E[log p(y* | x*, D)] = %0.3f', ...
                 mean(log_probabilities));
fprintf('%s\n', report);

subplot(2, 1, 2);
plot_predictions;
title(report);
