#!/usr/bin/env python3
"""
metaphone company deduplication (refactored)
uses metaphone phonetic similarity for company name matching
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
from shared_deduplication_utils import *

def main():
    print("=" * 60)
    print("METAPHONE DEDUPLICATION")
    print("=" * 60)
    
    # load data
    input_file = "companies_with_identical_coordinates.csv"
    if not os.path.exists(input_file):
        print(f"error: {input_file} not found")
        return
    
    df = load_and_prepare_data(input_file)
    
    # calculate metaphone similarity matrix
    print("calculating metaphone similarity matrix...")
    similarity_matrix = calculate_similarity_matrix(
        df['company_name_clean'], 
        metaphone_similarity, 
        threshold=0.8
    )
    
    # perform clustering
    threshold = 0.8
    cluster_labels = perform_clustering(similarity_matrix, threshold=threshold)
    
    # create results
    df_clustered, cluster_stats = create_cluster_dataframe(df, cluster_labels)
    canonical_df = create_canonical_companies(df_clustered, cluster_stats)
    
    # save results
    output_dir = "individual_output/new"
    clustered_file, canonical_file, analysis_file = save_results(
        df_clustered, canonical_df, cluster_stats, output_dir, "metaphone"
    )
    
    print(f"\nmetaphone deduplication complete!")
    print(f"found {len(cluster_stats)} clusters")
    print(f"results saved to {output_dir}")

if __name__ == "__main__":
    main() 