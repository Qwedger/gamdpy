import numpy as np
import numba
import math
from numba import cuda
import gamdpy as gp
from .interaction import Interaction

class TabulatedPairPotential(Interaction):
    """ Pair potential """

    def __init__(self, table_filename, params, max_num_nbs, exclusions=None):
        def params_function(i_type, j_type, params):
            result = params[i_type, j_type]            # default: read from params array
            return result            
    

        self.read_potential_file(table_filename)

        def pairpotential_function(ij_dist, ij_params, coefficients_array):
            Rmin, dr = ij_params[0], ij_params[1]
            loc = (ij_dist - Rmin)/dr
            index =  int(loc) # which interval this value of r is located
            two = numba.float32(2.0)
            three = numba.float32(3.0)
            six = numba.float32(6.0)

            if index < 0:
                index = 0
            elif index >= len(coefficients_array):
                index = len(coefficients_array) -1

            eps = loc - index # where in the given interval, r is located, goes from 0 to 1 (unless outside the range of the table)
            c = coefficients_array[index,:]
            v_interp = c[0] + eps * (c[1] + eps * (c[2] + eps * c[3]))
            v_prime = c[1] + eps * (two*c[2] + eps * three * c[3])

            s = - v_prime / ij_dist / dr # divide by dr to convert to derivative wrt r
            v_pp = (two*c[2] + six*c[3] * eps) / (dr*dr)  # divide by dr squared to convert to (second) derivative wrt r
            return v_interp, s, v_pp

        self.pairpotential_function = pairpotential_function
        self.params_function = params_function
        self.params_user = params
        self.exclusions = exclusions 
        self.max_num_nbs = max_num_nbs


    def read_single_table_from_file(self, pot_file, next_line):
        while next_line.startswith('#') or next_line == "\n":
            next_line = pot_file.readline()

        keyword = next_line.strip()
        params = pot_file.readline().split()
        assert params[0] == 'N'
        N = int(params[1])
        if params[2] != 'R' or len(params) != 5:
            raise ValueError('Only format N <N> R <rlo> <rhi> is supported')

        Rmin = float(params[3])
        Rmax = float(params[4])
        dr = (Rmax-Rmin)/(N-1)
        pot_table = np.zeros((N, 4))
        for idx in range(N):
            next_line = pot_file.readline()
            items = [float(x) for x in next_line.split()]
            pot_table[idx, :]  = np.array(items)

        return keyword, Rmin, dr, Rmax, pot_table

    def read_potential_file(self, filename):
        with open(filename, 'r') as potential_file:
            next_line = potential_file.readline()
            pot_tables = {}
            table_params = {}
            while next_line != "":
                label, Rmin, dr, Rmax, pot_table = self.read_single_table_from_file(potential_file, next_line)
                pot_tables[label] = pot_table
                table_params[label] = (Rmin, dr, Rmax)
                next_line = potential_file.readline()
        self.pot_tables = pot_tables
        self.table_params = table_params


    def generate_coefficients_array(self, dr, pot_table):
        """ This version can handle tables for different type pairs"""
        n_pts = len(pot_table) - 1
        coeffs = np.zeros((n_pts, 4), dtype=np.float32)
        #coeffs = np.zeros((n_pts, 4), dtype=np.float64)
        for index in range(n_pts):
            v0, v1 = pot_table[index:index+2,2]
            vp0, vp1 = pot_table[index:index+2,3] * dr
            coeffs[index, :] = v0, vp0, 3*(v1-v0) - 2*vp0 - vp1, 2*(v0-v1) + vp0 + vp1
        return coeffs


    def extract_params(self):
        """ This function extracts key params from data that was in the potential file"""

        num_params = 3
        num_types = len(self.params_user)
        params = np.zeros((num_types, num_types), dtype="f,"*num_params)
        all_coefficients = []
        cut_list = []
        for i in range(num_types):
            assert len(self.params_user[i]) == num_types
            these_coefficients = []
            for j in range(num_types):
                label = self.params_user[i][j]
                Rmin, dr, Rmax = self.table_params[label]
                params[i,j] = (Rmin, dr, Rmax)
                cut_list.append(Rmax)
                these_coefficients.append(self.generate_coefficients_array(dr, self.pot_tables[label]))
            all_coefficients.append(these_coefficients)


        max_cut = np.float32(max(cut_list))

        return params, all_coefficients, max_cut

    def evaluate_potential_function(self, r, types):
        params, all_coefficients, max_cut = self.extract_params()
        u, s, lap = self.pairpotential_function(r, params[types[0], types[1]], all_coefficients[0][0])
        return u


    def check_datastructure_validity(self) -> bool:
        nbflag = self.nblist.d_nbflag.copy_to_host()
        if nbflag[0] != 0 or nbflag[1] != 0:
            raise RuntimeError(f'Neighbor-list is invalid. Try allocating space for more neighbors (max_num_nbs in PairPot object). Allocated size: {self.max_num_nbs}, but {nbflag[1]+1} neighbours found. {nbflag=}.')
        return True


    def get_params(self, configuration: gp.Configuration, compute_plan: dict, verbose=False) -> tuple:

        self.params, self.all_coefficients, max_cut = self.extract_params()
        self.d_params = cuda.to_device(self.params)

        # make a two-dimensional tuple
        all_coefficients_list = []
        num_types = len(self.all_coefficients)
        for i in range(num_types):
            row_list = []
            for j in range(num_types):
                row_list.append(self.all_coefficients[i][j])
            all_coefficients_list.append(tuple(row_list)) # i'th item in main list is a tuple containing the different i,j tables for that particular i
        self.d_coefficients_array = cuda.to_device( tuple(all_coefficients_list) )



        if compute_plan['nblist'] == 'N squared':
            self.nblist = gp.NbList2(configuration, self.exclusions, self.max_num_nbs)
        elif compute_plan['nblist'] == 'linked lists':
            self.nblist = gp.NbListLinkedLists(configuration, self.exclusions, self.max_num_nbs)
        else:
            raise ValueError(f"No lblist called: {compute_plan['nblist']}. Use either 'N squared' or 'linked lists'")
        nblist_params = self.nblist.get_params(max_cut, compute_plan, verbose)

        return (self.d_params, self.nblist.d_nblist, nblist_params, self.d_coefficients_array)

    def get_kernel(self, configuration: gp.Configuration, compute_plan: dict, compute_flags: dict[str,bool], verbose=False):
        num_cscalars = configuration.num_cscalars

        compute_u = compute_flags['U']
        compute_w = compute_flags['W']
        compute_lap = compute_flags['lapU']
        compute_stresses = compute_flags['stresses']

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
        num_blocks = (num_part - 1) // pb + 1  

        if verbose:
            print(f'\tpb: {pb}, tp:{tp}, num_blocks:{num_blocks}')
            print(f'\tNumber (virtual) particles: {num_blocks*pb}')
            print(f'\tNumber of threads {num_blocks*pb*tp}')
            if compute_stresses:
                print('\tIncluding computation of stress tensor in pair potential')
        # Unpack indices for vectors and scalars to be compiled into kernel
        r_id, f_id = [configuration.vectors.indices[key] for key in ['r', 'f']]

        if compute_u:
            u_id = configuration.sid['U']
        if compute_w:
            w_id = configuration.sid['W']
        if compute_lap:
            lap_id = configuration.sid['lapU']

        if compute_stresses:
            sx_id = configuration.vectors.indices['sx']
            if D > 1:
                sy_id = configuration.vectors.indices['sy']
                if D > 2:
                    sz_id = configuration.vectors.indices['sz']
                    if D > 3:
                        sw_id = configuration.vectors.indices['sw']


        #pairpotential_function = self.pairpotential_function
        pairpotential_function = numba.njit(self.pairpotential_function)

        if UtilizeNIII:
            virial_factor_NIII = numba.float32( 1.0/configuration.D)
            #def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
            def pairpotential_calculator(ij_dist, ij_params, coefficients_array, dr, my_f, cscalars, my_stress, f, other_id):
                #u, s, umm = pairpotential_function(ij_dist, ij_params)
                u, s, umm = pairpotential_function(ij_dist, ij_params, coefficients_array)
                for k in range(D):
                    cuda.atomic.add(f, (other_id, k), dr[k]*s)
                    my_f[k] = my_f[k] - dr[k]*s                         # Force
                    if compute_w:
                        cscalars[w_id] += dr[k]*dr[k]*s*virial_factor_NIII  # Virial
                    if compute_stresses:
                        for k2 in range(D):
                            my_stress[k,k2] -= dr[k]*dr[k2]*s

                if compute_u:
                    cscalars[u_id] += u                                      # Potential energy
                if compute_lap:
                    cscalars[lap_id] += (numba.float32(1-D)*s + umm)*numba.float32( 2.0 ) # Laplacian 


                return
            
        else:
            virial_factor = numba.float32( 0.5/configuration.D )
            #def pairpotential_calculator(ij_dist, ij_params, dr, my_f, cscalars, my_stress, f, other_id):
            def pairpotential_calculator(ij_dist, ij_params, coefficients_array, dr, my_f, cscalars, my_stress, f, other_id):
                #u, s, umm = pairpotential_function(ij_dist, ij_params)
                u, s, umm = pairpotential_function(ij_dist, ij_params, coefficients_array)
                half = numba.float32(0.5)
                for k in range(D):
                    my_f[k] = my_f[k] - dr[k]*s                         # Force
                    if compute_w:
                        cscalars[w_id] += dr[k]*dr[k]*s*virial_factor       # Virial
                    if compute_stresses:
                        for k2 in range(D):
                            my_stress[k,k2] -= half*dr[k]*dr[k2]*s      # stress tensor
                if compute_u:
                    cscalars[u_id] += half*u                                # Potential energy
                if compute_lap:
                    cscalars[lap_id] += numba.float32(1-D)*s + umm          # Laplacian 
                return

        ptype_function = numba.njit(configuration.ptype_function)
        params_function = numba.njit(self.params_function)
        pairpotential_calculator = numba.njit(pairpotential_calculator)
        dist_sq_dr_function = numba.njit(configuration.simbox.get_dist_sq_dr_function())
    
        @cuda.jit( device=gridsync )  
        def calc_forces(vectors, cscalars, ptype, sim_box, nblist, params, coefficients_array):
            """ Calculate forces as given by pairpotential_calculator() (needs to exist in outer-scope) using nblist 
                Kernel configuration: [num_blocks, (pb, tp)]        
            """
            
            my_block = cuda.blockIdx.x
            local_id = cuda.threadIdx.x 
            global_id = my_block*pb + local_id
            my_t = cuda.threadIdx.y
            
            max_nbs = nblist.shape[1]-1            

            my_f = cuda.local.array(shape=D,dtype=numba.float32)
            my_dr = cuda.local.array(shape=D,dtype=numba.float32)
            my_cscalars = cuda.local.array(shape=num_cscalars, dtype=numba.float32)
            if compute_stresses:
                my_stress = cuda.local.array(shape=(D,D), dtype=numba.float32)
            else:
                my_stress = cuda.local.array(shape=(1,1), dtype=numba.float32)
        
            if global_id < num_part:
                for k in range(D):
                    #my_r[k] = r[global_id, k]
                    my_f[k] = numba.float32(0.0)
                    if compute_stresses:
                        for k2 in range(D):
                            my_stress[k,k2] = numba.float32(0.0)
                for k in range(num_cscalars):
                    my_cscalars[k] = numba.float32(0.0)
                my_type = ptype_function(global_id, ptype)
            
            cuda.syncthreads() # Make sure initializing global variables to zero is done

            if global_id < num_part:
                for i in range(my_t, nblist[global_id, max_nbs], tp):
                    other_id = nblist[global_id, i] 
                    other_type = ptype_function(other_id, ptype)
                    dist_sq = dist_sq_dr_function(vectors[r_id][other_id], vectors[r_id][global_id], sim_box, my_dr)
                    ij_params = params_function(my_type, other_type, params)
                    cut = ij_params[-1]
                    if dist_sq < cut*cut:
                        #pairpotential_calculator(math.sqrt(dist_sq), ij_params, my_dr, my_f, my_cscalars, my_stress, vectors[f_id], other_id)
                        pairpotential_calculator(math.sqrt(dist_sq), ij_params, coefficients_array[my_type][other_type], my_dr, my_f, my_cscalars, my_stress, vectors[f_id], other_id)
                for k in range(D):
                    cuda.atomic.add(vectors[f_id], (global_id, k), my_f[k])
                    if compute_stresses:
                        cuda.atomic.add(vectors[sx_id], (global_id, k), my_stress[0,k])
                        if D > 1:
                            cuda.atomic.add(vectors[sy_id], (global_id, k), my_stress[1,k])
                            if D > 2:
                                cuda.atomic.add(vectors[sz_id], (global_id, k), my_stress[2,k])
                                if D > 3:
                                    cuda.atomic.add(vectors[sw_id], (global_id, k), my_stress[3,k])

                for k in range(num_cscalars):
                    cuda.atomic.add(cscalars, (global_id, k), my_cscalars[k])

            return 
        
        nblist_check_and_update = self.nblist.get_kernel(configuration, compute_plan, compute_flags, verbose)

        if gridsync:
            # A device function, calling a number of device functions, using gridsync to syncronize
            @cuda.jit( device=gridsync )
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters, coefficients_array = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                grid.sync()
                calc_forces(vectors, scalars, ptype, sim_box, nblist, params, coefficients_array)
                return
            return compute_interactions
        
        else:
            # A python function, making several kernel calls to syncronize  
            def compute_interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
                params, nblist, nblist_parameters, coefficients_array = interaction_parameters
                nblist_check_and_update(grid, vectors, scalars, ptype, sim_box, nblist, nblist_parameters)
                calc_forces[num_blocks, (pb, tp)](vectors, scalars, ptype, sim_box, nblist, params, coefficients_array)
                return
            return compute_interactions


