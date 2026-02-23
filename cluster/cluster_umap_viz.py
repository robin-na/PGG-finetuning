"""
Visualize persona embeddings using UMAP
Load pre-computed embeddings from CSV and create UMAP visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import umap

# Configuration
EMBEDDINGS_CSV = "results/embeddings.csv"
OUTPUT_DIR = Path("results/umap_viz")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*80)
print("UMAP VISUALIZATION OF PERSONA EMBEDDINGS")
print("="*80)

# Step 1: Load embeddings from CSV
print(f"\n1. Loading embeddings from {EMBEDDINGS_CSV}...")
df = pd.read_csv(EMBEDDINGS_CSV)

# Extract persona IDs and embedding dimensions
persona_ids = df['persona_id'].values
embedding_cols = [col for col in df.columns if col.startswith('dim_')]
embeddings = df[embedding_cols].values

print(f"   ✓ Loaded {len(embeddings)} persona embeddings")
print(f"   ✓ Embedding dimensions: {embeddings.shape[1]}")

# Step 2: UMAP dimensionality reduction
print(f"\n2. Performing UMAP dimensionality reduction...")
print(f"   Parameters:")
print(f"     - n_neighbors: 15")
print(f"     - min_dist: 0.1")
print(f"     - metric: cosine")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='cosine',
    random_state=42
)

embeddings_2d = reducer.fit_transform(embeddings)

print(f"   ✓ UMAP completed")
print(f"   ✓ Reduced to 2D: shape {embeddings_2d.shape}")

# Step 3: Create visualization
print(f"\n3. Creating UMAP visualization...")

fig, ax = plt.subplots(figsize=(14, 10))

# Scatter plot
scatter = ax.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    c='steelblue',
    alpha=0.6,
    s=30,
    edgecolors='white',
    linewidths=0.5
)

ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
ax.set_title('UMAP Visualization of Persona Embeddings', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / 'umap_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved UMAP visualization: {output_path}")
plt.close()

# Step 4: Create density plot
print(f"\n4. Creating density plot...")

fig, ax = plt.subplots(figsize=(14, 10))

# 2D density plot using hexbin
hexbin = ax.hexbin(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    gridsize=50,
    cmap='YlOrRd',
    mincnt=1
)

ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
ax.set_title('UMAP Density Plot of Persona Embeddings', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('Density', fontsize=10, fontweight='bold')

plt.tight_layout()

# Save density plot
density_path = OUTPUT_DIR / 'umap_density.png'
plt.savefig(density_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved density plot: {density_path}")
plt.close()

# Step 5: Try different UMAP parameters
print(f"\n5. Creating UMAP visualizations with different parameters...")

param_sets = [
    {'n_neighbors': 5, 'min_dist': 0.01, 'name': 'tight_local'},
    {'n_neighbors': 30, 'min_dist': 0.5, 'name': 'loose_global'},
    {'n_neighbors': 50, 'min_dist': 0.1, 'name': 'large_neighborhood'},
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

# Original UMAP in first subplot
axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                c='steelblue', alpha=0.6, s=20, edgecolors='white', linewidths=0.3)
axes[0].set_title('Original (n_neighbors=15, min_dist=0.1)', fontweight='bold')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
axes[0].grid(True, alpha=0.3)

# Try different parameters
for idx, params in enumerate(param_sets, start=1):
    print(f"   - Testing: n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']}")

    reducer_test = umap.UMAP(
        n_neighbors=params['n_neighbors'],
        min_dist=params['min_dist'],
        n_components=2,
        metric='cosine',
        random_state=42
    )

    embeddings_test = reducer_test.fit_transform(embeddings)

    axes[idx].scatter(embeddings_test[:, 0], embeddings_test[:, 1],
                     c='steelblue', alpha=0.6, s=20, edgecolors='white', linewidths=0.3)
    axes[idx].set_title(f"{params['name']}\n(n_neighbors={params['n_neighbors']}, min_dist={params['min_dist']})",
                       fontweight='bold')
    axes[idx].set_xlabel('UMAP 1')
    axes[idx].set_ylabel('UMAP 2')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('UMAP with Different Parameters', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save comparison plot
comparison_path = OUTPUT_DIR / 'umap_parameter_comparison.png'
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved parameter comparison: {comparison_path}")
plt.close()

# Step 6: Save UMAP coordinates to CSV
print(f"\n6. Saving UMAP coordinates...")

umap_df = pd.DataFrame({
    'persona_id': persona_ids,
    'umap_x': embeddings_2d[:, 0],
    'umap_y': embeddings_2d[:, 1]
})

umap_csv_path = OUTPUT_DIR / 'umap_coordinates.csv'
umap_df.to_csv(umap_csv_path, index=False)
print(f"   ✓ Saved UMAP coordinates: {umap_csv_path}")
print(f"   ✓ CSV size: {len(umap_df)} rows × {len(umap_df.columns)} columns")

# Summary
print("\n" + "="*80)
print("UMAP VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nOutput files saved in: {OUTPUT_DIR}/")
print(f"  - umap_visualization.png: Main UMAP scatter plot")
print(f"  - umap_density.png: Density plot")
print(f"  - umap_parameter_comparison.png: Comparison of different parameters")
print(f"  - umap_coordinates.csv: UMAP coordinates for all personas")
print("="*80)
