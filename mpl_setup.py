import matplotlib.pyplot as plt
import matplotlib as mpl

# Set a nice font (you can adjust it to your preferred one)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['text.usetex'] = False #Disabled this: not working for macOS after pip install latex

# Set a larger font size for better readability
plt.rcParams['font.size'] = 12

# Ensure ticks appear on all four sides of the axes
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True

# Set ticks to be inside the plot
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Set tick size
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3

# Add minor ticks for more granularity
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# Set figure aesthetics
plt.rcParams['figure.figsize'] = [6.4, 4.8]  # Default figure size
plt.rcParams['axes.labelsize'] = 14  # Larger labels for axes
plt.rcParams['axes.titlesize'] = 16  # Larger title size
plt.rcParams['axes.linewidth'] = 1.5  # Thicker axes lines

# Set grid style to be subtle#
plt.rcParams['grid.alpha'] = 0.7  # Slight transparency for grid lines
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7

# Enable grid only for the major ticks
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.grid.which'] = 'major'

# Color and line style customization
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6

# Set default legend font size to 10
plt.rcParams['legend.fontsize'] = 10


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
