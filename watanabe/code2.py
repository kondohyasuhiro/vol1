import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
data=pd.read_csv("http://hosho.ees.hokudai.ac.jp/~kubo/stat/iwanamibook/fig/hbm/data7a.csv")

#種子の数yごとに集計、グラフとして表示すると
## from sum to count
plt.bar(range(9),data.groupby('y').count().id)
plt.show()
plt.savefig("output2/first.png")
data.groupby('y').count().T

#dataの制限
Y=np.array(data.y)[:100]

import numpy as np
import pymc3 as pm
import theano.tensor as T

def invlogit(v):
	return T.exp(v)/(T.exp(v)+1)

with pm.Model() as model_hier:
	s=pm.Uniform('s',lower=0,upper=1.0E+4)
	beta=pm.Normal('beta',0,1.0E-4)
	r=pm.Normal('r',0,tau=s,shape=len(Y))
	q=invlogit(beta+r)
	y=pm.Binomial('y',8,q,observed=Y)
	#step = pm.NUTS([s,beta,r])
	step = pm.Metropolis([s,beta,r])
	#step = pm.Slice([s,beta,r])
	trace_hier = pm.sample(10000, step)

with model_hier:
	pm.summary(trace_hier)
	pm.traceplot(trace_hier, model_hier.vars)
	pm.forestplot(trace_hier, varnames = ["s","beta"])
	#plt.savefig("output2/simple_linear_trace.png")

#with model_hier:
#	plt.bar(range(9), trace_hier[y]);
