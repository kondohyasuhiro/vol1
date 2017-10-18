import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

nikkei=pd.read_csv("https://www.quandl.com/api/v1/datasets/NIKKEI/INDEX.csv",
                                parse_dates={'Timestamp': ['Date']} ,
                                index_col='Timestamp')
nc=nikkei['Close Price'][:600]
returns=(np.log(nc).shift()-np.log(nc))

nc.plot(figsize=(30,15),title="Close Price of NIKKEI index")
plt.show()
#plt.savefig("output/code4-fig2.png")
returns.plot(title='return of NIKKEI index close price',figsize=(30,8))
plt.show()
nreturns=np.array(returns[1:])[::-1]

import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk
from scipy.sparse import csc_matrix
from scipy import optimize
import theano.tensor as T

with pm.Model() as model:
    #sigma, log_sigma = model.TransformedVar('sigma', pm.Exponential.dist(1./.02, testval=.1),
    #pm.logtransform)
    # TransformedVar was removed on 2015-6-13
    sigma = pm.Exponential('sigma', 1. / .02, testval=1)

    nu = pm.Exponential('nu', 1./10)
    s = GaussianRandomWalk('s', sigma**-2, shape=len(nreturns))
    r = pm.Exponential('r', nu, lam=np.exp(-2*s), observed=nreturns)

with model:
    start = pm.find_MAP(vars=[s], fmin=optimize.fmin_l_bfgs_b)
    step  = pm.NUTS(scaling=start)
    trace = pm.sample(2000, step, start,progressbar=False)

plt.plot(trace[s][::10].T,'b', alpha=.03)
plt.title('log volatility')


with model:
    pm.traceplot(trace, model.vars[:2])

exps=np.exp(trace[s][::10].T)
plt.plot(returns[:600][::-1])
plt.plot( exps, 'r', alpha=.03)
plt.plot(-exps, 'r', alpha=.03)
#plt.savefig("output/code4-fig1.png")
plt.show()
