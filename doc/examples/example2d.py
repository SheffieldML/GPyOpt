'''
Examples of use of the class BayesianOptimization
	- branin function

''' 
import GPy
import GPyOpt


np.random.seed(1234)

# create the object function
f_true = GPyOpt.fmodels.experiments2d.powers()
f_sim = GPyOpt.fmodels.experiments2d.powers(sd= 0.5)
#f_true.plot()
bounds = f_true.bounds
H = 1

# starts the optimization with 3 data points 
myBopt = GPyOpt.methods.BayesianOptimizationEI(bounds,acquisition_par=0.01)
myBopt.start_optimization(f_sim.f,H=H)
myBopt.plot_acquisition()
myBopt.plot_convergence()

# continue optimization for 10 observations more
myBopt.continue_optimization(H=10)
myBopt.plot_acquisition()
#myBopt.plot_convergence()
myBopt.suggested_sample












