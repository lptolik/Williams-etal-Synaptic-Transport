# If in Jupyter or within ipython run both following lines
# If in the terminal run nrnivmodl to load Neuron engine
#%%bash
#nrnivmodl

from neuron import h
import numpy as np
import pylab as plt
import scipy.linalg
from PyNeuronToolbox.record import ez_record,ez_convert
from PyNeuronToolbox.morphology import shapeplot,shapeplot_animate
from mpl_toolkits.mplot3d import Axes3D
from JSAnimation import IPython_display
from matplotlib import animation
from matplotlib.pyplot import cm
np.random.seed(123456789)

import sushi
import time

# Load morphology and other stuff
# --> SegLists: soma[2], dend[74], dend_5[37], apic[42], axon[1]
# --> Files from Migliore & Migliore (2012)
# --> CA1 pyramidal neuron
h.load_file('stdrun.hoc')
h.xopen('ri06.hoc')
h.xopen('fixnseg.hoc')
h.xopen('5a_nogui.hoc')
h.tstop = 700.0

from PyNeuronToolbox.morphology import dist_between,allsec_preorder

def dist_to_soma(segment):
    return dist_between(h,h.soma[0](0.5),segment)

#seglist in pre-order
sec_list = allsec_preorder(h)
seg_list = []
for sec in sec_list:
    locs = np.linspace(0,1,sec.nseg+2)[1:-1]
    for loc in locs:
        seg_list.append(sec(loc))
n = len(seg_list)

dts = [dist_to_soma(s) for s in seg_list]

initTime=time.time()
A,u,t,excess,err = sushi.run_uniform_reattachment(h, 0.1, 5, 50.0, 1e-8,1e-5)
FinalTime=time.time()-initTime
print(f"One function invocation {FinalTime}")

initTime=time.time()
A = sushi.make_uniform_reattachment_matrix(h, 0.1, 5, 50.0, 1e-8,1e-5)
FinalTime=time.time()-initTime
print(f"Make matrix {FinalTime}")
initTime=time.time()
u, t = sushi.run_sim(h, A)
FinalTime=time.time()-initTime
print(f"Simulate system only: {FinalTime}")
initTime=time.time()
u, t, excess, err = sushi.simulate_matrix(h, A)
FinalTime=time.time()-initTime
print(f"Simulate system with excess estimation: {FinalTime}")
