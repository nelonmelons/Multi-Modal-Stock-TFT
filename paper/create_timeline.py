"""
Create a simple experimental timeline diagram for the presentation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(12, 3))

# Define timeline
years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
positions = np.arange(len(years))

# Colors for different phases
train_color = '#4472C4'  # Blue
val_color = '#FFC000'    # Orange
test_color = '#C00000'   # Red

# Draw boxes for each phase
# Train: 2016-2019
train_box = FancyBboxPatch((0, 0.3), 3.9, 0.4, boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=train_color, linewidth=2)
ax.add_patch(train_box)
ax.text(2, 0.5, 'Training\n2016-2019', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')

# Val: 2020
val_box = FancyBboxPatch((4, 0.3), 0.9, 0.4, boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=val_color, linewidth=2)
ax.add_patch(val_box)
ax.text(4.5, 0.5, 'Val\n2020', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='black')

# Test: 2021-2024
test_box = FancyBboxPatch((5, 0.3), 3.9, 0.4, boxstyle="round,pad=0.05", 
                         edgecolor='black', facecolor=test_color, linewidth=2)
ax.add_patch(test_box)
ax.text(7, 0.5, 'Test\n2021-2024', ha='center', va='center', 
        fontsize=14, fontweight='bold', color='white')

# Add year markers
for i, year in enumerate(years):
    ax.text(i, 0.15, year, ha='center', va='top', fontsize=12)
    ax.plot([i, i], [0.25, 0.28], 'k-', linewidth=1)

# Draw timeline axis
ax.plot([0, len(years)-1], [0.25, 0.25], 'k-', linewidth=2)

# Add annotations
ax.annotate('Hyperparameter\nTuning', xy=(4.5, 0.75), xytext=(4.5, 1.1),
            ha='center', fontsize=11, 
            arrowprops=dict(arrowstyle='->', lw=1.5))

ax.annotate('Final\nEvaluation', xy=(7, 0.75), xytext=(7, 1.1),
            ha='center', fontsize=11, 
            arrowprops=dict(arrowstyle='->', lw=1.5))

# Add note about no look-ahead
ax.text(4.5, -0.1, 'Forward-only blocks\nNo look-ahead bias', 
        ha='center', va='top', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Set limits and remove axes
ax.set_xlim(-0.5, len(years)-0.5)
ax.set_ylim(-0.3, 1.3)
ax.axis('off')

plt.tight_layout()
plt.savefig('experiment_timeline.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Timeline diagram saved as 'experiment_timeline.png'")
plt.close()

# Create a second diagram showing the three horizons
fig, ax = plt.subplots(figsize=(10, 4))

# Draw time axis
ax.plot([0, 10], [2, 2], 'k-', linewidth=2)

# Current time
ax.plot([3, 3], [1.8, 2.2], 'k-', linewidth=3)
ax.text(3, 1.5, 't\n(today)', ha='center', fontsize=12, fontweight='bold')

# h=1
ax.annotate('', xy=(4, 2.5), xytext=(3, 2.5),
            arrowprops=dict(arrowstyle='<->', lw=2, color='blue'))
ax.text(3.5, 2.8, 'h=1 day', ha='center', fontsize=11, color='blue', fontweight='bold')

# h=5
ax.annotate('', xy=(5.5, 3), xytext=(3, 3),
            arrowprops=dict(arrowstyle='<->', lw=2, color='green'))
ax.text(4.25, 3.3, 'h=5 days\n(1 week)', ha='center', fontsize=11, color='green', fontweight='bold')

# h=21
ax.annotate('', xy=(8, 3.5), xytext=(3, 3.5),
            arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
ax.text(5.5, 3.8, 'h=21 days (1 month)', ha='center', fontsize=11, color='red', fontweight='bold')

# Add target points
ax.plot([4], [2], 'bo', markersize=10)
ax.plot([5.5], [2], 'go', markersize=10)
ax.plot([8], [2], 'ro', markersize=10)

# Labels
ax.text(4, 1.5, 't+1', ha='center', fontsize=10)
ax.text(5.5, 1.5, 't+5', ha='center', fontsize=10)
ax.text(8, 1.5, 't+21', ha='center', fontsize=10)

# Title
ax.text(3, 4.3, 'Multi-Horizon Prediction Task', ha='center', fontsize=14, fontweight='bold')
ax.text(3, 0.8, 'Target: Log returns  $r_{t→t+h} = \\ln(P_{t+h}) - \\ln(P_t)$', 
        ha='center', fontsize=11, style='italic')

ax.set_xlim(1, 10)
ax.set_ylim(0.5, 4.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('multi_horizon.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Multi-horizon diagram saved as 'multi_horizon.png'")
plt.close()

print("\nDiagrams created successfully!")
print("Move them to the paper/figs directory if needed.")
