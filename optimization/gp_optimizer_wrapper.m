function [nlZ, dnlZ, HnlZ] = gp_optimizer_wrapper(hyperparameter_values, ...
          prototype_hyperparameters, inference_method, mean_function, ...
          covariance_function, likelihood, x, y)

  hyperparameters = rewrap(prototype_hyperparameters, ...
                           hyperparameter_values(:));

  if (nargout <= 1)
    [~, nlZ] = inference_method(hyperparameters, mean_function, ...
            covariance_function, likelihood, x, y);

    return;

  elseif (nargout == 2)
    [~, nlZ, dnlZ] = inference_method(hyperparameters, mean_function, ...
            covariance_function, likelihood, x, y);

  elseif (nargout > 2)
    [~, nlZ, dnlZ, HnlZ] = inference_method(hyperparameters, mean_function, ...
            covariance_function, likelihood, x, y);

    HnlZ = HnlZ.H;
  end

  dnlZ = unwrap(dnlZ);

end