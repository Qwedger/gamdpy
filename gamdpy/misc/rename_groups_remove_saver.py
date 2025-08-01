## This is an helper script to rename group names in .h5 output
import h5py

def rename_group_traj(filename):
    ''' Group 'trajectory_saver' -> 'trajectory' '''
    with h5py.File(filename, "r+") as fin:
        if 'trajectory_saver' in fin.keys():
            fin['trajectory'] = fin['trajectory_saver']
            del fin['trajectory_saver']
        else:
            print(".h5 has already group trajectory")

def rename_group_stress(filename):
    ''' Group 'stress_saver' -> 'stresses' '''
    with h5py.File(filename, "r+") as fin:
        if 'stress_saver' in fin.keys():
            fin['stresses'] = fin['stress_saver']
            del fin['stress_saver']
        else:
            print(".h5 has already group stresses")

def rename_group_scalar(filename):
    ''' Group 'scalar_saver' -> 'scalars' '''
    with h5py.File(filename, "r+") as fin:
        if 'scalar_saver' in fin.keys():
            fin['scalars'] = fin['scalar_saver']
            del fin['scalar_saver']
        else:
            print(".h5 has already group scalars")

def group_to_dataset(filename):
    ''' Remove group structure for group scalars containing a single dataset '''
    with h5py.File(filename, "r+") as fin:
        if isinstance(fin['scalars'], h5py.Group):
            print("Group")

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
