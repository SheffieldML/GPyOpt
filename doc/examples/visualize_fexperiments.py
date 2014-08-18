''' 
Script to visualize the available (2d) test functions in GPyOpt
 
Note that each created object contains.
 - *.f : the funtion itself (can be evaluated at any point)
 - *.plot: a 3d plot of the function if the dimension.
 - *.min : value of the global minimum(s) for the default parameters.
 - *.fmin: the value of the function at the minimum
 - *.bouds: box domain of the function.

Javier Gonzalez August, 2014
'''


import GPyOpt

## (1) schaffer function
schaffer = GPyOpt.fmodels.experiments2d.schaffer2()
schaffer.plot()
schaffer.min
schaffer.fmin
schaffer.bounds

## (2) levy2 function
levy2 = GPyOpt.fmodels.experiments2d.levy2()
levy2.plot()
levy2.min
levy2.fmin
levy2.bounds

## (3) mccormick function
mccormick = GPyOpt.fmodels.experiments2d.mccormick()
mccormick.plot()
mccormick.min
mccormick.fmin
mccormick.bounds

## (4) sixhumpcamel function
sixhumpcamel = GPyOpt.fmodels.experiments2d.sixhumpcamel()
sixhumpcamel.plot()
sixhumpcamel.min
sixhumpcamel.fmin
sixhumpcamel.bounds

## (5) beale function
beale = GPyOpt.fmodels.experiments2d.beale()
beale.plot()
beale.min
beale.fmin
beale.bounds

## (6) goldstein function
goldstein = GPyOpt.fmodels.experiments2d.goldstein()
goldstein.plot()
goldstein.min
goldstein.fmin
goldstein.bounds

## (7) eggholder function
eggholder = GPyOpt.fmodels.experiments2d.eggholder()
eggholder.plot()
eggholder.min
eggholder.fmin
eggholder.bounds

## (8) dropwave function
dropwave = GPyOpt.fmodels.experiments2d.dropwave(bounds = [(-2,2),(-2,2)])
dropwave.plot()
dropwave.min
dropwave.fmin
dropwave.bounds

## (9) cross-in-Trai functions
crossintray = GPyOpt.fmodels.experiments2d.crossintray()
crossintray.plot()
crossintray.min
crossintray.fmin
crossintray.bounds

## (10) branin function
branin = GPyOpt.fmodels.experiments2d.branin()
branin.plot()
branin.min
branin.fmin
branin.bounds


