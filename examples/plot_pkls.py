""" 
Generate a python script which plots data stored in pickle files, 
as generated eg. by analyze_dynamics.py and analyze_structure.py 

Examples:
---------
Assume you have simulations of the Kob&Andersen binary mixture at different temperatures, 
and you have run analyze_dynamics.py and analyze_structure.py on all of them.

> python3 plot_pkls.py times msd type=0 loglog Data/KABLJ_Rho1.200_T*_dynamics.pkl | python3
Makes a log-log plot of the mean-square displacement of the A (type=0) particles stored in
the files fullfilling the pattern 'Data/KABLJ_Rho1.200_T*_dynamics.pkl'.

Note that plot_pkls.py actually prints a python script to the terminal, which in the example above
is piped to python3, which in turn makes the plot. This means that if you want to alter the apperence
of the plot, you can pipe the script to a file instead ( python3 plot_pkls ... > myscript.py),
and work from there.

Plot the intermediate scattering function of the B-particles
> python3 plot_pkls.py times Fs type=1 loglin Data/KABLJ_Rho1.200_T*_dynamics.pkl | python3

Plot the A-B radial distribution function
python3 plot_pkls.py distances rdf type=0,1  Data/KABLJ_Rho1.200_T*_rdf.pkl | python3

"""

import pickle
import sys

import matplotlib.pyplot as plt
print('import matplotlib.pyplot as plt')
print('import pickle')

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
xkey = argv.pop(0)
ykey = argv.pop(0)

plotfunc = "plt.plot"
type_index = '0'

filenames = []
for entry in argv:
    if entry[:5] == 'type=':
        type_index = entry[5:]
    if entry == 'loglog':
        plotfunc = 'plt.loglog'
    if entry == 'linlog':
        plotfunc = 'plt.semilogy'
    if entry == 'loglin':
        plotfunc = 'plt.semilogx'
    if entry[-4:] == ".pkl":
        filenames.append(entry)

print('\nfilenames = ', filenames)
print('\nlegends = ', filenames)

with open(filenames[0],'rb') as f:
    data = pickle.load(f)
    print(f'\n# Available keys in {filenames[0]}:', data.keys())

print("\nplt.figure(figsize=(10,6))")
print("for filename, legend in zip(filenames, legends):")
print("\t with open(filename,'rb') as f:")
print("\t\t data = pickle.load(f)")
print("\t", f"{plotfunc}(data['{xkey}'], data['{ykey}'][:,{type_index}], '.-', label=filename)" )
print("plt.legend()")
print(f"plt.xlabel('{xkey}')")
print(f"plt.ylabel('{ykey}')")
print("plt.grid(linestyle='--', alpha=0.5)")
print("plt.show()")
