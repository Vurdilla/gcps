import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Plot matrix of particle concentration
def process_matrixpc_file(filename, figname, minlog10level, maxlog10level, blockSize, levelRescale, saveDPI, printColorbar, printAxes, printNames):
    with open(filename, 'r') as file:
        content = file.readlines()
    nstrings = len(content)
    #print(content)
    print(nstrings)
    
    # Extract number of tables
    readstring = 0
    plotname = content[readstring].strip()
    readstring += 1
    snum_table, snum_rows, snum_cols = content[readstring].strip().split('\t')
    readstring += 1
    num_tables = int(snum_table)
    num_cols = int(snum_cols)  # Number of columns for subplots
    num_rows = int(snum_rows)  # Number of rows for subplots
    print('nTables ' + str(num_tables))
    set_name, table_sizenameX, table_sizenameY = content[readstring].strip().split('\t')
    readstring += 1
    table_sizeX = int(table_sizenameX)
    table_sizeY = int(table_sizenameY)
    print('set_name ' + str(set_name))
    print('table_size ' + str(table_sizeX) + ' x ' + str(table_sizeY))
    
    # Initialize plot
    
    if num_tables > num_rows * num_cols:
        num_cols += 1

    print(str(num_tables) + ': ' + str(num_cols) + ' x ' + str(num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    if num_tables > 1:
        axes = axes.flatten()

    # Plot each table
    boxSize = 1
    readstring += 1
    for table_num in range(num_tables):
        # if table_num % (everyCol * everyRow) == 0:
        print('Parsing table ' + str(table_num))
        table_name = content[readstring].strip()
        readstring += 1
        print('Table name ' + str(table_name))
        table_subname = content[readstring].strip()
        readstring += 1
        ctrlstring, boxSizeName = content[readstring].strip().split('\t')
        boxSize = float(boxSizeName)
        print('boxSize ' + str(boxSize))
        readstring += 1

        # if table_num % (everyCol * everyRow) == 0:
        table_data = np.genfromtxt(filename, skip_header=readstring, max_rows=table_sizeY+1)
        read_rows = table_data.shape[0]
        read_cols = table_data.shape[1]
        print('Read table ' + str(read_cols) + ' x ' +str(read_rows))
        # print(table_data)
        x = table_data[1:, 0]  # Assuming first column contains x values, excluding the first row
        y = table_data[0, 1:]  # Assuming first row contains y values, excluding the first column
        f = table_data[1:, 1:]  # Assuming the remaining data is Z values for the function
        num_blocks_x = f.shape[0] // blockSize
        num_blocks_y = f.shape[1] // blockSize
        f_binned = np.zeros((num_blocks_x, num_blocks_y), dtype=f.dtype)
        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                f_binned[i, j] = np.log10(np.sum(f[i*blockSize:(i+1)*blockSize, j*blockSize:(j+1)*blockSize]))
        # Bin x-axis: take mean of each block along x
        x_binned = np.array([np.mean(x[i*blockSize:(i+1)*blockSize]) for i in range(num_blocks_x)])
        
        # Bin y-axis: take mean of each block along y
        y_binned = np.array([np.mean(y[j*blockSize:(j+1)*blockSize]) for j in range(num_blocks_y)])
        
        np.log10(f_binned)
        #log10f = np.log10(f_binned)
    
        if num_tables > 1:
            ax = axes[table_num]
        else:
            ax = axes

        xmin = np.min(x)
        xmax = np.max(x)
        ymin = np.min(y)
        ymax = np.max(y)
        min_nonzero = np.min(f_binned[f_binned != 0])
        max_nonzero = np.max(f_binned[f_binned != 0])
        x_shift = x - boxSize*0.5
        y_shift = y - boxSize*0.5
        xbinned_shift = x_binned - boxSize*0.5
        ybinned_shift = y_binned - boxSize*0.5
        #cmin = np.log10f(min_nonzero)
        #cmax = np.log10f(max_nonzero)
        #print('cmin '+str(cmin))
        #print('cmax '+str(cmax))
        f_sum = np.sum(f)
        f_binned_sum = np.sum(f_binned)
        #log10f_sum = np.sum(log10f)
        print('Sum of f:', f_sum)
        print('Sum of f_binned:', f_binned_sum)
        #print('Sum of log10f:', log10f_sum)
        cmin = minlog10level
        cmax = maxlog10level 
        if levelRescale == True:
            cmax = maxlog10level * (table_num + 1) / num_tables
        #im = ax.imshow(f_binned, interpolation='bilinear', cmap='turbo', extent=[0, 1, 0, 1], vmin=cmin, vmax=cmax)
        values_mask = np.isnan(f_binned) | (f_binned < cmin)
        X, Y = np.meshgrid(x_binned, y_binned, indexing='ij')
        distance = np.sqrt((X-boxSize*0.5)**2 + (Y-boxSize*0.5)**2)
        print('distance min '+str(np.min(distance)))
        print('distance max '+str(np.max(distance)))
        circle_mask = distance <= boxSize*0.5
        final_mask = circle_mask & values_mask
        f_binned[final_mask] = cmin
        #cmap = cm.get_cmap('turbo').copy()  # copy so we can modify it
        #cmap.set_bad(cmap(0.0))  # set background (bad/masked/nan) to lowest color
        im = ax.imshow(f_binned.T, interpolation='bilinear', cmap='turbo', extent=[x_shift.min(), x_shift.max(), y_shift.min(), y_shift.max()], vmin=cmin, vmax=cmax, origin='lower')
        #if table_num % num_cols == num_cols - 1:
        if printColorbar > 0:
            divider = make_axes_locatable(ax) # Create a divider to add a colorbar without changing the plot size
            cax = divider.append_axes("right", size="10%", pad=0.1)  # Control the colorbar's size and position
            colorbar = plt.colorbar(im, cax=cax) # Add the colorbar to the custom axis
        #colorbar = plt.colorbar(im, ax=ax, shrink=0.65, pad=0.1)
        # colorbar.ColorbarBase(ax, cmap='binary', norm=colors.Normalize(vmin=cmin, vmax=cmax))
        if printAxes > 0:
            ax.set_xlabel('')  # Set x-axis label
            ax.set_ylabel('')  # Set y-axis label
            ax.set_xlim(-boxSize*0.5, boxSize*0.5)
            ax.set_ylim(-boxSize*0.5, boxSize*0.5)
            ax.set_xticks([-boxSize*0.5, -boxSize*0.25, 0 , boxSize*0.25, boxSize*0.5])
            ax.set_yticks([-boxSize*0.5, -boxSize*0.25, 0 , boxSize*0.25, boxSize*0.5])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.grid(False)
        if printNames > 0:
            title = table_name + '\n' + table_subname
            ax.set_title(title)  # Set subplot title
    
        readstring += table_sizeY + 2  # Move to the next table in content
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(figname, format='png', dpi=saveDPI)
    print('Done successfully')


nargs = len(sys.argv) - 1
args = sys.argv[1:]  # Exclude the script name

mpcFileName = args[0]
mpcOutName = args[1]
pminPC = 0.0
pmaxPC = 2.0
blockSizePC = 2
pminPC = float(args[2])
pmaxPC = float(args[3])
blockSizePC = int(args[4])
saveDPI = float(args[5])
printColorbar = int(args[6])
printAxes = int(args[7])
printNames = int(args[8])

process_matrixpc_file(mpcFileName, mpcOutName, pminPC, pmaxPC, blockSizePC, False, saveDPI, printColorbar, printAxes, printNames)
