import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from pymc3.backends.base import merge_traces
import theano.tensor as T

saveimage=False

disasters_data = np.array([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
years = len(disasters_data)
plt.plot(disasters_data,".-")
if(saveimage):
    plt.savefig("img1.png")

with pm.Model() as model_disaster:
    switchpoint = pm.DiscreteUniform('switchpoint', lower=0, upper=years)
    early_mean = pm.Exponential('early_mean', lam=1.)
    late_mean = pm.Exponential('late_mean', lam=1.)
    idx = np.arange(years)
    rate = T.switch(switchpoint >= idx, early_mean, late_mean)
    disasters = pm.Poisson('disasters', rate, observed=disasters_data)

n=1000
with model_disaster:
	start = {'early_mean': 2., 'late_mean': 3.}
	step1 = pm.Slice([early_mean, late_mean])
	step2 = pm.Metropolis([switchpoint])
	trace_disaster = pm.sample(n, tune=500, start=start, step=[step1, step2],progressbar=False)

with model_disaster:
    pm.summary(trace_disaster)
    pm.traceplot(trace_disaster,model_disaster.vars)
    if(saveimage):
        plt.savefig("c3-trace.png")

#結果の重ね書き
#xrange is for python 2.6
plt.show()
with model_disaster:
    for i in range(len(trace_disaster[early_mean])):
        e=trace_disaster[early_mean][i]
        l=trace_disaster[late_mean][i]
        s=trace_disaster[switchpoint][i]
        v=[ e if(y<s) else l for y in range(years)]
        plt.plot(range(years),v,alpha=0.03,c='blue')
        if(saveimage):
            plt.saveimage("c3-a.png")
