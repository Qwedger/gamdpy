import numba

def apply_cubic_spline_cutoff(pair_potential):  
    # Cut-off by computing potential twice, avoiding changes to params
    """ Apply cubic spline cutoff to a pair-potential function

   

    Note: calls original potential function  twice, avoiding changes to params

    Parameters
    ----------
        pair_potential: callable
            a function that calculates a pair-potential:
            u, s, umm =  pair_potential(dist, params)

    Returns
    -------

        potential: callable
            a function where shifted force cutoff is applied to original function

    """
    pair_pot = numba.njit(pair_potential)

    @numba.njit
    def potential(dist, params): # pragma: no cover
        cut_outer = params[-1]
        cut_inner = params[-2]
        Delta_r = cut_outer - cut_inner


        u_bare, s_bare, umm_bare = pair_pot(dist, params)



        #u_cut_outer, s_cut_outer, umm_cut_outer = pair_pot(cut_outer, params)
        u_cut_inner, s_cut_inner, umm_cut_inner = pair_pot(cut_inner, params)

        C1, C2, C3, C4 = (s_cut_inner*cut_inner,
                          -umm_cut_inner,
                             (-3*s_cut_inner*cut_inner + 2*umm_cut_inner * Delta_r) / Delta_r**2,
                             (2*s_cut_inner*cut_inner - umm_cut_inner*Delta_r)/Delta_r**3)
        # for the potential, we integrate, and include a constant C0 whose value is such that the potential is zero at the outer cutoff
        C0 = - (C1*Delta_r + C2*Delta_r**2/2 + C3*Delta_r**3/3 + C4*Delta_r**4/4)
        # And now we have to shift the LJ potential inside the inner cutoff to match the potential at that point
        K = -C0 - u_cut_inner
    

        if dist < cut_inner:
            s = s_bare
            u = u_bare + K
            umm = umm_bare
        else:
            delta_r = dist - cut_inner
            u = - (C0 + C1*delta_r + C2*delta_r**2/2 + C3*delta_r**3/3 + C4*delta_r**4/4)
            s = (C1 + C2*delta_r + C3*delta_r**2 + C4*delta_r**3)/dist
            umm = - (C2 + 2*C3*delta_r + 3*C4*delta_r**2)


        return u, s, umm

    return potential

