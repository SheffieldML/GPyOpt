'''
Examples of use of the class BayesianOptimization
	- branin function

''' 
import GPyOpt


# create the object function
f_true = GPyOpt.fmodels.experiments2d.branin()
f_sim = GPyOpt.fmodels.experiments2d.branin(sd= .5)
f_true.plot()
bounds = f_true.bounds
H = 50

# starts the optimization with 3 data points 
myBopt = GPyOpt.methods.BayesianOptimization(bounds, acquisition_type='MPI', acquisition_par = 0.01)
myBopt.start_optimization(f_sim.f,H=H)
myBopt.plot_acquisition()
myBopt.plot_convergence()

for k in range(10):
# cotinue optimization for 10 observations more
	myBopt.continue_optimization(H=50)
	myBopt.plot_acquisition()
	myBopt.plot_convergence()
	myBopt.suggested_sample
	f_true.min
	f_true.fmin











