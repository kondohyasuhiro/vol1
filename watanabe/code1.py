import matplotlib.pyplot as plt
import numpy as np

multicore=True
saveimage=False

N=40
X=np.random.uniform(10,size=N)
Y=X*30+4+np.random.normal(0,16,size=N)
plt.plot(X,Y,"o")
if(saveimage):
	plt.savefig("code1-fig1.png")

import pymc3 as pm
import time
from pymc3.backends.base import merge_traces


itenum=1000
t0=time.clock()
chainnum=3


with pm.Model() as model:
	alpha = pm.Normal('alpha', mu=0, sd=20)
	beta = pm.Normal('beta', mu=0, sd=20)
	# default upper=1
	sigma = pm.Uniform('sigma', lower=0, upper = 40)
	y = pm.Normal('y', mu=beta*X + alpha, sd=sigma, observed=Y)
	start = pm.find_MAP()
	step = pm.NUTS()

with model:
	if(multicore):
		trace = pm.sample(itenum, step, start=start,
					njobs=chainnum, random_seed=range(chainnum), progress_bar=False)
	else:
		ts=[pm.sample(itenum, step, chain=i, progressbar=False) for i in range(chainnum)]
		trace=merge_traces(ts)

	pm.traceplot(trace)
	if(saveimage):
		plt.savefig("c1-simple_linear_trace.png")
	print ("Rhat="+str(pm.gelman_rubin(trace)))

t1=time.clock()
print ("elapsed time="+str(t1-t0))

#trace
if(not multicore):
	trace=ts[0]
with model:
	pm.traceplot(trace,model.vars)
	if(saveimage):
		plt.savefig("c1-trace.png")
	pm.forestplot(trace)
	if(saveimage):
		plt.savefig("c1-forest.png")
	pm.summary(trace)
"""
import pickle as pkl
with open("simplelinearregression_model.pkl","w") as fpw:
	pkl.dump(model,fpw)
with open("simplelinearregression_trace.pkl","w") as fpw:
	pkl.dump(trace,fpw)
with open("simplelinearregression_model.pkl") as fp:
	model=plk.load(fp)
with open("simplelinearregression_trace.pkl") as fp:
	trace=plk.load(fp)
"""
