'''
Examples of use of the class BayesianOptimization
	- branin function

''' 
import GPyOpt


# create the object function
f_true = GPyOpt.fmodels.experiments2d.branin()
f_sim = GPyOpt.fmodels.experiments2d.branin(sd= .5)
#f_true.plot()
bounds = f_true.bounds
H = 5

# starts the optimization with 3 data points 
myBopt = GPyOpt.methods.BayesianOptimizationEI(bounds)
myBopt.start_optimization(f_sim.f,H=H)
myBopt.plot_acquisition()
myBopt.plot_convergence()

a =myBopt._init_model(myBopt.X,myBopt.Y)


for k in range(10):
# cotinue optimization for 10 observations more
	myBopt.continue_optimization(H=50)
	myBopt.plot_acquisition()
	myBopt.plot_convergence()
	myBopt.suggested_sample
	f_true.min
	f_true.fmin











