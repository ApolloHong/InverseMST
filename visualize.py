import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

# ==========================================
# Style Configuration (IEEE/ACM Paper Style)
# ==========================================
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (14, 10),
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

# ==========================================
# Data Processing
# ==========================================
try:
    df = pd.read_csv("advanced_benchmark.csv")
except FileNotFoundError:
    print("Error: CSV not found. Run the C++ benchmark first.")
    exit()

# Metric for X-axis: Problem Size (Nodes * Edges is a good proxy for sparse graphs)
df['Size'] = df['N'] * df['M']
df = df.sort_values('Size')

# ==========================================
# Theoretical Fitting Functions
# ==========================================
def poly_n2(x, a, b): return a * x + b           # O(Size) ~ O(MN)
def poly_n3(x, a, b): return a * x**1.5 + b      # O(Size^1.5)
def poly_n4(x, a, b): return a * x**2 + b        # O(Size^2)

# ==========================================
# Plotting
# ==========================================
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# ------------------------------------------
# Plot 1: Cost Comparison (Optimality Gap)
# ------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
# Filter only MCMF and Greedy for visual clarity
subset = df[df['Algorithm'].isin(['MCMF', 'Greedy'])].copy()
# Create a pivot table to compare side-by-side
pivot_df = subset.pivot(index='Test', columns='Algorithm', values='Cost')
pivot_df.plot(kind='bar', ax=ax1, color=['#e74c3c', '#2ecc71'], alpha=0.9, width=0.8)

ax1.set_title("(a) Optimality Analysis: Greedy vs MCMF", fontweight='bold')
ax1.set_ylabel("Total Modification Cost ($L_1$ Norm)")
ax1.set_xlabel("Test Case ID")
ax1.legend(title="Algorithm")
# Annotate Greedy Failure
ax1.text(0.5, 0.9, "Greedy overestimates\ncost significantly", transform=ax1.transAxes, 
         ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# ------------------------------------------
# Plot 2: Time Complexity Scaling (Log-Log)
# ------------------------------------------
ax2 = fig.add_subplot(gs[0, 1])
colors = {'MCMF': '#2ecc71', 'Greedy': '#e74c3c', 'General_LP': '#3498db', 'Ellipsoid': '#9b59b6'}
markers = {'MCMF': 'o', 'Greedy': 'x', 'General_LP': 's', 'Ellipsoid': '^'}

for algo in df['Algorithm'].unique():
    data = df[df['Algorithm'] == algo]
    ax2.plot(data['Size'], data['Time_ms'], label=algo, 
             color=colors[algo], marker=markers[algo], linestyle='None')
    
    # Fit Curve (Visualizing the complexity class)
    if len(data) > 2:
        popt, _ = curve_fit(poly_n2, data['Size'], data['Time_ms'])
        x_fit = np.linspace(data['Size'].min(), data['Size'].max(), 100)
        
        # Select curve based on theory
        if algo == 'MCMF':
            # MCMF is efficient, fit linear-ish to Size
            popt, _ = curve_fit(poly_n2, data['Size'], data['Time_ms'])
            y_fit = poly_n2(x_fit, *popt)
        elif algo == 'Ellipsoid':
             # Steep curve
            popt, _ = curve_fit(poly_n4, data['Size'], data['Time_ms'])
            y_fit = poly_n4(x_fit, *popt)
        else:
            popt, _ = curve_fit(poly_n3, data['Size'], data['Time_ms'])
            y_fit = poly_n3(x_fit, *popt)
            
        ax2.plot(x_fit, y_fit, color=colors[algo], linestyle='--', alpha=0.5, linewidth=1.5)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title("(b) Runtime Scalability (Log-Log Scale)", fontweight='bold')
ax2.set_ylabel("Execution Time (ms)")
ax2.set_xlabel("Problem Size ($N \\times M$)")
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.2)

# ------------------------------------------
# Plot 3: Performance Profile (Dolan-More style)
# ------------------------------------------
ax3 = fig.add_subplot(gs[1, :])
# Normalize time against the best algorithm for each test case
best_times = df.groupby('Test')['Time_ms'].min()
df['Ratio'] = df.apply(lambda row: row['Time_ms'] / best_times[row['Test']], axis=1)

for algo in df['Algorithm'].unique():
    data = df[df['Algorithm'] == algo].sort_values('Ratio')
    y = np.arange(1, len(data) + 1) / len(data)
    # Append end point for step plot
    x_plot = np.concatenate(([0], data['Ratio'], [data['Ratio'].max()*1.1]))
    y_plot = np.concatenate(([0], y, [1]))
    
    ax3.step(x_plot, y_plot, where='post', label=algo, color=colors[algo])

ax3.set_xlim(1, 100) # Limit x-axis to see the fast algorithms
ax3.set_xscale('log')
ax3.set_title("(c) Performance Profile (Probability of solving within x times optimal speed)", fontweight='bold')
ax3.set_xlabel("Performance Ratio ($\\tau$)")
ax3.set_ylabel("Probability ($P(r_{p,s} \\leq \\tau$)")
ax3.legend(loc="lower right")

# Save
plt.savefig("Analysis_Report.png", dpi=300, bbox_inches='tight')
print("Visualization generated: Analysis_Report.png")
plt.show()