from ..core.bo import BO


class ModularBayesianOptimization(BO):
    def __init__(self, model, space, objective, acquisition, evaluator, X_init, Y_init=None, cost = None, normalize_Y = True, model_update_interval = 1):

        # --- Create optimization space
        super(ModularBayesianOptimization ,self).__init__(  model                  = model, 
                                                            space                  = space, 
                                                            objective              = objective, 
                                                            acquisition            = acquisition, 
                                                            evaluator              = evaluator,
                                                            X_init                 = X_init, 
                                                            Y_init                 = Y_init,
                                                            cost                   = cost,
                                                            normalize_Y            = normalize_Y, 
                                                            model_update_interval  = model_update_interval)

        # --- Initilize everyting
        self.run_optimization(0)