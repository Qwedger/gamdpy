import numba

def apply_cubic_spline_cutoff(pair_potential):  
    # Cut-off by computing potential twice, avoiding changes to params
    """ Apply cubic spline cutoff to a pair-potential function

    Actually the cubic spline is applied to the pair force as done by LAMMPS
    (see https://docs.lammps.org/pair_lj_smooth.html).
     The potential energy is given by a quartic function in the transition region.It is shifted
     in each region to ensure its continuity at the inner and outer cutoffs.

    Note: calls original potential function  twice each time, avoiding changes to params.
    The four coefficients of the spline are also computed each time, along with the two energy
    shift constants, which results in a small but noticeable performance cost.

    Parameters
    ----------
        pair_potential: callable
            a function that calculates a pair-potential:
            u, s, umm =  pair_potential(dist, params)

    Returns
    -------

        potential: callable
            a function where a cubic spline cutoff is applied to original function

    """
    pair_pot = numba.njit(pair_potential)

    @numba.njit
    def potential(dist, params): # pragma: no cover
        cut_outer = params[-1]
        cut_inner = params[-2]
        Delta_r = cut_outer - cut_inner


        u_bare, s_bare, umm_bare = pair_pot(dist, params)
        u_cut_inner, s_cut_inner, umm_cut_inner = pair_pot(cut_inner, params)

        two = numba.float32(2.0)
        three = numba.float32(3.0)
        four = numba.float32(4.0)

        C1, C2, C3, C4 = (s_cut_inner*cut_inner,
                          -umm_cut_inner,
                             (-three*s_cut_inner*cut_inner + two*umm_cut_inner * Delta_r) / Delta_r**2,
                             (two*s_cut_inner*cut_inner - umm_cut_inner*Delta_r)/Delta_r**3)
        # for the potential, we integrate, and include a constant C0 whose value is such that the potential is zero at the outer cutoff
        C0 = - (C1*Delta_r + C2*Delta_r**2/two + C3*Delta_r**3/three + C4*Delta_r**4/four)
        # And now we have to shift the LJ potential inside the inner cutoff to match the potential at that point
        K = -C0 - u_cut_inner


        if dist < cut_inner:
            s = s_bare
            u = u_bare + K
            umm = umm_bare
        else:
            delta_r = dist - cut_inner
            u = - (C0 + C1*delta_r + C2*delta_r**2/two + C3*delta_r**3/three + C4*delta_r**4/four)
            s = (C1 + C2*delta_r + C3*delta_r**2 + C4*delta_r**3)/dist
            umm = - (C2 + two*C3*delta_r + three*C4*delta_r**2)


        return u, s, umm

    return potential

