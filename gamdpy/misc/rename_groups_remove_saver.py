## This is an helper script to rename group names in .h5 output
# Files produced with SHA preciding 4d499b1d3bedabf513d50f9843dbf04c501aa71a should be updated

import h5py

def rename_group_traj(filename):
    ''' Rename group 'trajectory_saver' -> 'trajectory' '''
    with h5py.File(filename, "r+") as fin:
        if 'trajectory_saver' in fin.keys():
            fin['trajectory'] = fin['trajectory_saver']
            del fin['trajectory_saver']
        else:
            print(".h5 has already group trajectory")

def rename_group_stress(filename):
    ''' Rename group 'stress_saver' -> 'stresses' '''
    with h5py.File(filename, "r+") as fin:
        if 'stress_saver' in fin.keys():
            fin['stresses'] = fin['stress_saver']
            del fin['stress_saver']
        else:
            print(".h5 has already group stresses")

def rename_group_scalar(filename):
    ''' Rename group 'scalar_saver' -> 'scalars' '''
    with h5py.File(filename, "r+") as fin:
        if 'scalar_saver' in fin.keys():
            fin['scalars'] = fin['scalar_saver']
            del fin['scalar_saver']
        else:
            print(".h5 has already group scalars")

# This function is not used for now
def group_to_dataset(filename, group_name):
    ''' Remove group structure for a group containing a single dataset '''
    print(f"Demoting group {group_name} with a single dataset to dataset")
    with h5py.File(filename, "r+") as fin:
        if not isinstance(fin[group_name], h5py.Group):
            print(f"{group_name} is not a group")
            return
        elif len(fin[group_name].keys())!=1:
            print(f"{group_name} has more than one dataset, doing nothing")
            print(fin[group_name].keys())
            return
        else:
            dataset_key = [key for key in fin[group_name].keys()][0]
            if len(fin[f'{group_name}/{dataset_key}'].attrs)!=0:
                print(f"{group_name}/{dataset_key} has attrs but should not")
                exit()
            fin['backup'] = fin[group_name]
            del fin[group_name]
            fin[group_name] = fin[f'backup/{dataset_key}']
            for key, val in fin['backup'].attrs.items():
                fin[group_name].attrs[key] = val
            del fin['backup']

def rename_all(filename):
    rename_group_traj(filename)
    rename_group_stress(filename)
    rename_group_scalar(filename)

if __name__ == "__main__":
    import sys
    if len(sys.argv)!=2:
        print("Script takes .h5 as input")
    elif not sys.argv[1].endswith(".h5"):
        print("Script takes .h5 as input")
    else:
        filename = sys.argv[1]
        print(f"Updating group names in {filename}")
    rename_all(filename)
    #group_to_dataset(filename, 'scalars')
    #group_to_dataset(filename, 'stresses')
