"""Generate a clean comparison table image for the loss shape results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

# Data from the runs
headers = ['Shape', 'Spectral FoM (%)', 'R_back (thin)', 'R_back (thick)']
rows = [
    ['sKK (HT derivative)', '97.3', '0.02239', '0.00003'],
    ['Gaussian',             '82.7', '0.02203', '0.00059'],
    ['Constant',             '0.0',  '0.02393', '0.00116'],
    ['Random',               '0.0',  '0.02372', '0.00444'],
    ['Double peaks',         '0.0',  '0.02349', '0.01337'],
]

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis('off')

table = ax.table(
    cellText=rows,
    colLabels=headers,
    loc='center',
    cellLoc='center',
)

table.auto_set_font_size(False)
table.set_fontsize(14)

# Style
for (row, col), cell in table.get_celld().items():
    cell.set_fontsize(14)
    cell.set_text_props(fontfamily='Arial')
    cell.set_edgecolor('#cccccc')
    cell.set_height(0.14)
    if row == 0:
        # Header row
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold', fontfamily='Arial')
    elif row % 2 == 0:
        cell.set_facecolor('#D6E4F0')
    else:
        cell.set_facecolor('white')

# Highlight the sKK row (row 1) with a subtle green tint
for col in range(len(headers)):
    table[(1, col)].set_facecolor('#E2EFDA')

# Add subtitle below table
ax.text(0.5, -0.02,
        'Thin: k_steep=100, d ≈ 0.4 μm, ∫ε″dx = 0.238    |    '
        'Thick: k_steep=10, d ≈ 4.0 μm, ∫ε″dx = 2.384',
        transform=ax.transAxes, fontsize=11, ha='center', va='top',
        fontfamily='Arial', color='#555555')

table.auto_set_column_width(col=list(range(len(headers))))

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig_shape_comparison_table.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {FIGDIR}/fig_shape_comparison_table.png')
