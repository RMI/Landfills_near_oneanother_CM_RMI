#%%

import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd

filepath = 'climate_trace_sites.csv'
df = pd.read_csv(filepath)
#df = df[df['Data Source'] == 'Climate Trace']
df = df[df['gas'] == 'ch4']
df.reset_index(inplace=True, drop=True)

# Extract longitude and latitude
df['Longitude'] = df['st_astext'].str.extract(r'POINT\((.*?)\s', expand=False).astype(float)
df['Latitude'] = df['st_astext'].str.extract(r'\s(.*?)\)', expand=False).astype(float)

# Drop the original column
df.drop('st_astext', axis=1, inplace=True)

#%%

# Convert Lat/Lon to radians for use in haversine formula
trace_rad = np.deg2rad(df[['Latitude', 'Longitude']])

# Construct BallTree
TRACE_tree = BallTree(trace_rad, metric='haversine')

# Query the tree for sites within 3 km
indices, distances = TRACE_tree.query_radius(trace_rad, r=3/6371, return_distance=True)

matches = []
for i, (distance, index) in enumerate(zip(distances, indices)):
    for d, idx in zip(distance, index):
        matches.append({
            'lat1': df.at[i, 'Latitude'],
            'lon1': df.at[i, 'Longitude'],
            'id1': df.at[i, 'asset_id'],
            'idx1': i,
            'lat2': df.at[idx, 'Latitude'],
            'lon2': df.at[idx, 'Longitude'],
            'id2': df.at[idx, 'asset_id'],
            'idx2': idx,
            'distance': d * 6371  # Convert radians to km
        })

matches = pd.DataFrame(matches)

# %%

# Get rid of sites that matched with themselves
matches_df = matches[matches['idx1'] != matches['idx2']].copy()

# Sort the matches by the first site's index
#matches_df.sort_values(by='idx1', inplace=True)

# Get all the site IDs that matched with at least one other site
sites_near_sites = set(matches_df['id1'].unique()).union(set(matches_df['id2'].unique()))
print(len(matches_df['id1'].unique()))
print(len(matches_df['id2'].unique()))
print(len(sites_near_sites))
# As expected, sites_near_sites, matches_df['id1'].unique(), and matches_df['id2'].unique() are all the same length

# Get a subset of the original df that only contains sites that matched
subset_df = df[df['asset_id'].isin(sites_near_sites)].copy()
print(len(subset_df))

subset_df['data_source'] = 'Climate TRACE'
subset_df = subset_df[['asset_id', 'data_source', 'asset_name', 'asset_type', 'Latitude', 'Longitude']]
subset_df.rename(columns={"asset_id": "source_asset_id"}, inplace=True)

subset_df.to_csv('waste_sites_3km.csv', index=False)

# %%
