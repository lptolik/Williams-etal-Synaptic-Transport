# Download NEURON: http://www.neuron.yale.edu/neuron/download
# Download PyNeuronToolbox: https://github.com/ahwillia/PyNeuron-Toolbox

from __future__ import division
from neuron import h
import numpy as np
from scipy.linalg import expm  # matrix exponential used to solve linear system
import pylab as plt
from matplotlib import animation
from matplotlib.pyplot import cm
from copy import copy
from PyNeuronToolbox import morphology

## Get a list of segments
from PyNeuronToolbox.morphology import shapeplot, allsec_preorder, root_indices, dist_between, shapeplot_animate


def dist_to_soma(segment):
    return dist_between(h, h.soma[0](0.5), segment)


def get_nsegs(h):
    N = 0
    for sec in h.allsec():
        N += sec.nseg
    return int(N)


def sushi_system(h, ar, br, cr, dr, ap, bp, cp, dp, t, l):
    """
    Returns a matrix A, such that dx/dt = A*x

    N = # of compartments
    A is (4N x 4N) matrix
    x is (4N x 1) vector.
      The first N elements correspond to concentrations of r (RNA molecules in transit)
      The second N elements correspond to concentrations of r-star (detached/active RNA molecules)
      The third N elements correspond to concentrations of p (protein molecules in transit)
      The last N elements correspond to concentrations of p-star (detached/active protein molecules)
    The trafficking rate constants along the microtubules are given by the vectors "a" and "b"
    The rate constants for u detaching/attaching (turning into u*) are given by "c" and "d"
    The rate constants for transcription are given by "t"
    The rate constants for protein degradation are given by "l"
    """

    N = len(dr)
    sec_list = allsec_preorder(h)

    ## State-space equations
    #  dx/dt = Ax + Bu
    A = np.zeros((4 * N, 4 * N))

    # Trafficking along belt
    # Iterative traversal of dendritic tree in pre-order
    i = 0
    section = None
    parentStack = [(None, sec_list[0])]
    while len(parentStack) > 0:
        # Get next section to traverse
        #  --> p is parent index, section is h.Section object
        (p, section) = parentStack.pop()

        # Trafficking to/from parent
        if p is not None:
            # RNA Out of parent, into child
            ai = ar.pop()
            A[p, p] += -ai
            A[i, p] += ai
            # RNA Into parent, out of child
            bi = br.pop()
            A[p, i] += bi
            A[i, i] += -bi
            # Protein Out of parent, into child
            ai = ap.pop()
            A[2 * N + p, 2 * N + p] += -ai
            A[2 * N + i, 2 * N + p] += ai
            # Into parent, out of child
            bi = bp.pop()
            A[2 * N + p, 2 * N + i] += bi
            A[2 * N + i, 2 * N + i] += -bi

        # visit all segments in compartment
        for (j, seg) in enumerate(section):
            # Deal with out/into rates within compartment, just tridiag matrix
            if j > 0:
                # RNA Out of parent, into child
                ai = ar.pop()
                A[i - 1, i - 1] += -ai
                A[i, i - 1] += ai
                # RNA Into parent, out of child
                bi = br.pop()
                A[i - 1, i] += bi
                A[i, i] += -bi
                # Protein Out of parent, into child
                ai = ap.pop()
                A[2 * N + i - 1, 2 * N + i - 1] += -ai
                A[2 * N + i, 2 * N + i - 1] += ai
                # Protein Into parent, out of child
                bi = bp.pop()
                A[2 * N + i - 1, 2 * N + i] += bi
                A[2 * N + i, 2 * N + i] += -bi
            # move onto next compartment
            i += 1

        # now visit children in pre-order
        child_list = list(h.SectionRef(sec=section).child)
        if child_list is not None:
            child_list.reverse()
        for c_sec in child_list:
            parentStack.append([i - 1, c_sec])  # append parent index and child

    # Detachment off the belt
    for i in range(N):
        # RNA detachment
        A[i, i] += -cr[i]
        A[i + N, i] += cr[i]
        # Protein detachment
        A[2 * N + i, 2 * N + i] += -cp[i]
        A[3 * N + i, 2 * N + i] += cp[i]

    # Reattachment to belt
    for i in range(N):
        # RNA reattachment
        A[i, i + N] += dr[i]
        A[i + N, i + N] += -dr[i]
        # Protein reattachment
        A[2 * N + i, 3 * N + i] += dp[i]
        A[3 * N + i, 3 * N + i] += -dp[i]

    # Protein synthesis
    for i in range(N):
        # RNA translation
        A[3 * N + i, i + N] += t[i]
        # Protein degradation
        A[3 * N + i, 3 * N + i] += -l[i]

    return A


def set_uniform_rates(h, diff_coeff):
    """
    (a+b) = 2 * diff_coeff / (dist_between(p,i)**2)
    (c+d) = 2 * diff_coeff * compartment_size
    """
    a, b, c, d = [], [], [], []
    sec_list = allsec_preorder(h)

    # Iterative traversal of dendritic tree in pre-order
    i = 0
    parentStack = [(None, None, sec_list[0])]
    while len(parentStack) > 0:
        # Get next section to traverse
        #  --> p is parent index, section is h.Section object
        (p, psize, section) = parentStack.pop()
        segsize = section.L / section.nseg

        # Trafficking to/from parent
        if p is not None:
            a.insert(0, diff_coeff / ((0.5 * (psize + segsize)) ** 2))
            b.insert(0, diff_coeff / ((0.5 * (psize + segsize)) ** 2))

        # visit all segments in section
        for (j, seg) in enumerate(section):
            # detachment and reattachment
            c.insert(0, diff_coeff * segsize)
            d.insert(0, diff_coeff * segsize)

            # trafficking rates within compartment, just tridiag matrix
            if j > 0:
                a.insert(0, diff_coeff / (segsize ** 2))
                b.insert(0, diff_coeff / (segsize ** 2))

            # move onto next compartment
            i += 1

        # now visit children in pre-order
        child_list = list(h.SectionRef(sec=section).child)
        if child_list is not None:
            child_list.reverse()  # needed to visit children in correct order
        for c_sec in child_list:
            parentStack.append((i - 1, segsize, c_sec))  # append parent index and child

    return a, b, c, d


def run_uniform_reattachment(h, dscale, diff_coeff_r, diff_coeff_p, l, t, **kwargs):
    """
    Make system matrix and simulate it
    :param h: neuron geometry
    :param dscale: how faster/slower detachment compare to attachment
    :param diff_coeff_r: rna diffusion coefficient
    :param diff_coeff_p: protein diffusion coefficient
    :param l: protein degradation rate
    :param t: protein translation rate
    :param kwargs: other parameters to pass to the run_sim function
    :return: A -- system matrix, u -- system solution, t -- solution time
              excess and err -- error
    """
    # get trafficking rates (no reattachment)
    A = make_uniform_reattachment_matrix(h, dscale, diff_coeff_p, diff_coeff_r, l, t)
    u, t, excess, err = simulate_matrix(h, A, **kwargs)

    return A, u, t, list(excess), err


def make_uniform_reattachment_matrix(h, dscale, diff_coeff_p, diff_coeff_r, l, t):
    """
    Make system matrix
    :param h: neuron geometry
    :param diff_coeff_r: rna diffusion coefficient
    :param diff_coeff_p: protein diffusion coefficient
    :param l: protein degradation rate
    :param t: protein translation rate
    :return: A -the system matrix
    """

    ar, br, cr, _ = set_uniform_rates(h, diff_coeff_r)
    dr = [ci * dscale for ci in cr]
    ap, bp, cp, _ = set_uniform_rates(h, diff_coeff_p)
    dp = [ci * dscale for ci in cp]
    tl = [t for ci in cp]
    ll = [l for ci in cp]
    # get state-transition matrix
    A = sushi_system(h, ar, br, cr, dr, ap, bp, cp, dp, tl, ll)
    return A


def simulate_matrix(h, A, **kwargs):
    """
    Simulate system definid by the matrix and estimate excess and error
    :param h: neuron geometry
    :param A: the system matrix
    :param kwargs: other parameters to pass to the run_sim function
    :return: u -- system solution, t -- solution time
              excess and err -- error
    """
    N = get_nsegs(h)
    u, t = run_sim(h, A, **kwargs)
    # calculate excess % of cargo left on microtuble
    total_cargo = np.sum(u[0, :])
    excess = 100 * np.sum(u[:, :N], axis=1) / total_cargo
    # calculate error
    targ = np.sum(u[0, :]) / N
    err = 100 * np.mean(np.abs(u[:, N:] - targ) / targ, axis=1)
    return u, t, excess, err


def run_uniform_sim(h, cscale, diff_coeff, **kwargs):
    # get trafficking rates (no reattachment)
    N = get_nsegs(h)
    a, b, _, _ = set_uniform_rates(h, diff_coeff)
    c = list(np.ones(N) * cscale)

    # get state-transition matrix
    A = sushi_system(h, a, b, c)
    u, t = run_sim(h, A, **kwargs)

    # calculate error
    targ = np.sum(u[0, :]) / N
    err = 100 * np.mean(np.abs(u[:, N:] - targ) / targ, axis=1)
    final_err = u[-1, N:] - targ

    return A, u, t, list(err), final_err


def run_sim(h, A, t0=2e1, tmax=5e7, dt=2):
    u0 = np.zeros(A.shape[0])
    N = float(A.shape[0] / 4)
    roots = root_indices(allsec_preorder(h))
    for r in roots:
        u0[r] = 1.0 / len(roots)
    u = [u0, np.dot(expm(t0 * A), u0)]
    t = [0, t0]
    while t[-1] < tmax:
        t.append(t[-1] * dt)
        u.append(np.dot(expm(t[-1] * A), u0))

    return np.array(u), np.array(t)


def save_movie(h, t, u, view, filename, clim=[0, 2]):
    # Make an animation
    fig = plt.figure(figsize=(8, 8))
    shapeax = plt.subplot(111, projection='3d')
    lines = shapeplot(h, shapeax, order='pre', lw=2)
    shapeax.view_init(*view)
    plt.title('cargo distribution over (log) time', fontweight='bold', fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(clim[0], clim[1]))
    sm._A = []
    plt.colorbar(sm, shrink=0.5)
    shapeax.set_axis_off()
    plt.tight_layout()

    anim = None

    # anim_func = shapeplot_animate(u,lines,clim=clim,cmap=cm.cool)
    # anim = animation.FuncAnimation(fig, anim_func, frames=u.shape[0], interval=400, blit=True)
    # anim.save('./anim/'+filename+'.mp4', fps=30)
    return anim


def calc_tradeoff_curve(h, diff_coeff=10.0):
    N = get_nsegs(h)
    a, b, c, _ = set_uniform_rates(h, diff_coeff)
    y = np.ones(N) / N
    u0 = np.zeros(N * 2)
    u0[0] = 1.0

    tau, err = [], []
    tss = 0  # initial lower bound
    for detach_ts in np.logspace(-2, -6, 20):
        c = list(np.ones(N) * detach_ts)
        A = sushi_system(h, copy(a), copy(b), c)
        tss = calc_time_to_ss(A, u0, lower_bound=tss / 2)
        uss = np.dot(expm(A * 10 * tss), u0)  # steady-state profile
        err.append(100 * np.mean(np.abs((y - uss[N:]) / y)))
        tau.append(tss / 60)

    return np.array([[t, er] for t, er in zip(tau, err)])


def calc_tradeoff_reattachment(h, diff_coeff=10.0):
    N = get_nsegs(h)
    a, b, c, _ = set_uniform_rates(h, diff_coeff)
    y = np.ones(N) / N
    u0 = np.zeros(N * 2)
    u0[0] = 1.0

    tau, excess = [], []
    tss = 0  # initial lower bound
    for ds in np.flipud(np.logspace(2, -2, 10)):
        d = [ci * ds for ci in c]
        A = sushi_system(h, copy(a), copy(b), c, d)
        tss = calc_time_to_ss_reattachment(A, u0, lower_bound=tss / 2)
        uss = np.dot(expm(A * 10 * tss), u0)  # steady-state profile
        excess.append(100 * (np.sum(uss[:N]) / np.sum(uss)))
        tau.append(tss / 60)
        if excess[-1] > 90: break

    return np.array([[t, exc] for t, exc in zip(tau, excess)])


def calc_time_to_ss(A, u0, lower_bound=0, perc_ss=0.1, tol=1.0):
    """ Calculate number of seconds to reach steady-state (within perc_ss)
    """
    N = int(len(u0) / 2)
    np.testing.assert_approx_equal(np.sum(u0), 1.0)
    upper_bound = 1e10
    while (upper_bound - lower_bound) > tol:
        tt = lower_bound + (upper_bound - lower_bound) / 2
        u = np.dot(expm(A * tt), u0)
        if np.sum(u[:N]) > perc_ss:
            # not converged to steady-state
            lower_bound = tt
        else:
            # converged to within perc_ss of steady-state
            upper_bound = tt
    return lower_bound + (upper_bound - lower_bound) / 2


def calc_time_to_ss_reattachment(A, u0, perc_ss=0.1, bound_tol=1.0, lower_bound=0):
    """ Calculate number of seconds to reach steady-state (within perc_ss)
    """
    N = get_nsegs(h)
    upper_bound = 1e10

    uss = np.dot(expm(A * upper_bound), u0)
    sum_uss = np.sum(uss[N:])

    tt = 1.0
    u = np.dot(expm(A * tt), u0)
    upper_bound, lower_bound = 1e10, 0.0
    while (upper_bound - lower_bound) > bound_tol:
        tt = lower_bound + (upper_bound - lower_bound) / 2
        u = np.dot(expm(A * tt), u0)
        if np.mean(np.abs(u[N:] - uss[N:]) / uss[N:]) > perc_ss:
            # not converged to steady-state
            lower_bound = tt
        else:
            # converged to steady-state
            upper_bound = tt
    tss = lower_bound + (upper_bound - lower_bound) / 2
    return tss


def snapshots(h, u, t, folder, cellname, view, u_cmap, us_cmap):
    xsc = np.array([0, 100])
    ysc = np.array([0, 0])
    N = int(u.shape[1] / 4)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(*view)

    for i in xrange(len(t)):
        plt.cla()
        ax.set_axis_off()
        morphology.shapeplot(h, ax, clim=[0, 2], cvals=u[i, :N], cmap=u_cmap)
        # ax.plot(xsc,ysc,'-r',lw=2)
        # ax.plot(ysc,xsc,'-r',lw=2)
        plt.title(t[i])
        plt.savefig('./' + folder + '/' + cellname + '_u_t_' + str(int(i)) + '.eps')

        plt.cla()
        ax.set_axis_off()
        morphology.shapeplot(h, ax, clim=[0, 2], cvals=u[i, N:], cmap=us_cmap)
        # ax.plot(xsc,ysc,'-r',lw=2)
        # ax.plot(ysc,xsc,'-r',lw=2)
        plt.title(t[i])
        plt.savefig('./' + folder + '/' + cellname + '_us_t_' + str(int(i)) + '.eps')

    plt.close()


def calculate_difference(h, fitdf, u, t, delta=1, useAmount=True):
    """
    Calculates eucledian distance between data in pandas data frame and model solution
    :param h: neuron topology
    :param fitdf: experimental data, two columns X for coordinates and V for values
    :param u: model solution from run_sim
    :param t: model solution times from run_sim
    :return: series of the same length as t with distances between solution and data
    """
    seg_idx = prepare_seg_index(h,fitdf,delta=delta)
    return make_dist_calc(seg_idx, u, useAmount)


def make_dist_calc(seg_idx, u, fitdt, useAmount):
    resp = np.zeros((fitdt.shape[0], u.shape[0]))
    for i in range(seg_idx.shape[0]):
        ii = int(seg_idx[i, 0])
        ij = int(seg_idx[i, 1])
        if useAmount:
            resp[ii] += u[:, ij].T * seg_idx[i, 2]
        else:
            resp[ii] += u[:, ij].T
    dist = [sum((resp[:, i] - fitdt['V']) ** 2) for i in range(resp.shape[1])]
    return dist


def prepare_seg_index(h,fitdt,delta=1):
    sec_list = allsec_preorder(h)
    seg_list = []
    for sec in sec_list:
        locs = np.linspace(0, 1, sec.nseg + 2)[1:-1]
        for loc in locs:
            seg_list.append(sec(loc))
    n = len(seg_list)
    ss=seg_list[419].volume()

    dts = [dist_to_soma(s) for s in seg_list]
    seg_idx = [(i, j,seg_list[j].volume()) for i in range(fitdt.shape[0]) for j in range(n) if abs(dts[j] - fitdt['X'][i]) < 1]
    return np.array(seg_idx)

