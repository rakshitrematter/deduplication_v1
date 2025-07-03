#!/usr/bin/env python3
"""
location-based company deduplication (refactored)
uses geographic coordinates for company clustering
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
from shared_deduplication_utils import *

def location_similarity_matrix(df, lat_threshold=0.01, lon_threshold=0.01):
    """calculate location similarity matrix between companies"""
    n = len(df)
    similarity_matrix = np.zeros((n, n))
    
    print("calculating location similarity matrix...")
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            similarity = location_similarity(df.iloc[i], df.iloc[j], lat_threshold, lon_threshold)
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    
    return similarity_matrix

def main():
    print("=" * 60)
    print("LOCATION-BASED DEDUPLICATION")
    print("=" * 60)
    
    # load data
    input_file = "companies_with_identical_coordinates.csv"
    if not os.path.exists(input_file):
        print(f"error: {input_file} not found")
        return
    
    df = load_and_prepare_data(input_file)
    
    # calculate location similarity matrix
    similarity_matrix = location_similarity_matrix(df)
    
    # perform clustering
    threshold = 0.8
    cluster_labels = perform_clustering(similarity_matrix, threshold=threshold)
    
    # create results
    df_clustered, cluster_stats = create_cluster_dataframe(df, cluster_labels)
    canonical_df = create_canonical_companies(df_clustered, cluster_stats)
    
    # save results
    output_dir = "individual_output/new"
    clustered_file, canonical_file, analysis_file = save_results(
        df_clustered, canonical_df, cluster_stats, output_dir, "location_based"
    )
    
    print(f"\nlocation-based deduplication complete!")
    print(f"found {len(cluster_stats)} clusters")
    print(f"results saved to {output_dir}")

if __name__ == "__main__":
    main() 