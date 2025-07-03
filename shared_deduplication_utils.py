import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import re
import os
from tqdm import tqdm

def load_and_prepare_data(file_path):
    """load and prepare company data from csv file"""
    print(f"loading data from {file_path}")
    df = pd.read_csv(file_path)
    print(f"loaded {len(df)} companies")
    
    # clean company names
    df['company_name_clean'] = df['company_name'].str.lower().str.strip()
    df['company_name_clean'] = df['company_name_clean'].apply(lambda x: re.sub(r'[^\w\s]', '', x) if pd.notna(x) else x)
    
    return df

def calculate_similarity_matrix(names, similarity_func, threshold=0.8):
    """calculate similarity matrix between company names"""
    n = len(names)
    similarity_matrix = np.zeros((n, n))
    
    print("calculating similarity matrix...")
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            if pd.notna(names[i]) and pd.notna(names[j]):
                similarity = similarity_func(names[i], names[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
    
    return similarity_matrix

def perform_clustering(similarity_matrix, threshold=0.8, min_samples=2):
    """perform clustering using dbscan on similarity matrix"""
    print("performing clustering...")
    
    # convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # perform dbscan clustering
    clustering = DBSCAN(eps=1-threshold, min_samples=min_samples, metric='precomputed')
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    return cluster_labels

def create_cluster_dataframe(df, cluster_labels):
    """create dataframe with cluster information"""
    df_clustered = df.copy()
    df_clustered['cluster_id'] = cluster_labels
    
    # calculate cluster statistics
    cluster_stats = []
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:  # noise points
            continue
            
        cluster_companies = df_clustered[df_clustered['cluster_id'] == cluster_id]
        cluster_stats.append({
            'cluster_id': cluster_id,
            'size': len(cluster_companies),
            'companies': cluster_companies['company_name'].tolist(),
            'company_ids': cluster_companies['company_id'].tolist()
        })
    
    return df_clustered, cluster_stats

def create_canonical_companies(df_clustered, cluster_stats):
    """create canonical companies dataframe"""
    canonical_companies = []
    
    for cluster in cluster_stats:
        # select representative company (first one in cluster)
        representative = df_clustered[df_clustered['cluster_id'] == cluster['cluster_id']].iloc[0]
        
        canonical_companies.append({
            'canonical_company_id': f"canonical_{cluster['cluster_id']}",
            'canonical_company_name': representative['company_name'],
            'cluster_id': cluster['cluster_id'],
            'cluster_size': cluster['size'],
            'member_company_ids': ','.join(map(str, cluster['company_ids'])),
            'member_company_names': '; '.join(cluster['companies'])
        })
    
    return pd.DataFrame(canonical_companies)

def save_results(df_clustered, canonical_df, cluster_stats, output_dir, method_name):
    """save clustering results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # save clustered companies
    clustered_file = os.path.join(output_dir, f"{method_name}_clustered.csv")
    df_clustered.to_csv(clustered_file, index=False)
    print(f"saved clustered companies to {clustered_file}")
    
    # save canonical companies
    canonical_file = os.path.join(output_dir, f"{method_name}_canonical.csv")
    canonical_df.to_csv(canonical_file, index=False)
    print(f"saved canonical companies to {canonical_file}")
    
    # save cluster analysis
    analysis_file = os.path.join(output_dir, f"{method_name}_analysis.txt")
    with open(analysis_file, 'w') as f:
        f.write(f"cluster analysis for {method_name}\n")
        f.write("=" * 50 + "\n\n")
        
        total_clusters = len(cluster_stats)
        total_companies = sum(cluster['size'] for cluster in cluster_stats)
        
        f.write(f"total clusters: {total_clusters}\n")
        f.write(f"total companies in clusters: {total_companies}\n\n")
        
        # cluster size distribution
        cluster_sizes = [cluster['size'] for cluster in cluster_stats]
        f.write("cluster size distribution:\n")
        for size in sorted(set(cluster_sizes)):
            count = cluster_sizes.count(size)
            f.write(f"  {size} companies: {count} clusters\n")
        
        f.write("\ncluster details:\n")
        for cluster in cluster_stats:
            f.write(f"\ncluster {cluster['cluster_id']} ({cluster['size']} companies):\n")
            for company in cluster['companies']:
                f.write(f"  - {company}\n")
    
    print(f"saved cluster analysis to {analysis_file}")
    
    return clustered_file, canonical_file, analysis_file

def fuzzy_jaccard_similarity(str1, str2):
    """calculate fuzzy jaccard similarity between two strings"""
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    
    # tokenize strings
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # calculate fuzzy intersection
    intersection = 0
    for token1 in tokens1:
        for token2 in tokens2:
            if fuzz.ratio(token1, token2) >= 80:  # 80% similarity threshold
                intersection += 1
                break
    
    # calculate fuzzy union
    union = len(tokens1) + len(tokens2) - intersection
    
    return intersection / union if union > 0 else 0.0

def metaphone_similarity(str1, str2):
    """calculate metaphone similarity between two strings"""
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    
    from metaphone import doublemetaphone
    
    # get metaphone codes
    meta1, _ = doublemetaphone(str1)
    meta2, _ = doublemetaphone(str2)
    
    if not meta1 or not meta2:
        return 0.0
    
    # calculate similarity
    return fuzz.ratio(meta1, meta2) / 100.0

def jaro_winkler_similarity(str1, str2, threshold=0.85):
    """calculate jaro-winkler similarity between two strings"""
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    
    similarity = fuzz.jaro_winkler_similarity(str1, str2)
    return similarity if similarity >= threshold else 0.0

def location_similarity(row1, row2, lat_threshold=0.01, lon_threshold=0.01):
    """calculate location-based similarity between two companies"""
    try:
        lat1, lon1 = float(row1['latitude']), float(row1['longitude'])
        lat2, lon2 = float(row2['latitude']), float(row2['longitude'])
        
        # check if coordinates are within threshold
        lat_diff = abs(lat1 - lat2)
        lon_diff = abs(lon1 - lon2)
        
        if lat_diff <= lat_threshold and lon_diff <= lon_threshold:
            # calculate distance-based similarity
            distance = np.sqrt(lat_diff**2 + lon_diff**2)
            max_distance = np.sqrt(lat_threshold**2 + lon_threshold**2)
            similarity = 1 - (distance / max_distance)
            return max(similarity, 0.8)  # minimum 80% similarity for nearby locations
        
        return 0.0
        
    except (ValueError, TypeError):
        return 0.0 