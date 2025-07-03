# Company Deduplication Project - Comprehensive Documentation

## Overview
This project implements multiple company deduplication methods using fuzzy matching and clustering algorithms. The goal is to identify and group similar companies from a dataset, creating canonical representations for each cluster.

## Project Structure

### Core Files
- **`shared_deduplication_utils.py`** - Shared utilities module containing common functions used across all deduplication methods
- **`code-ensemble_loc.py`** - Ensemble location-based deduplication combining name and location similarity
- **`code-fuzzy_jaccard_refactored.py`** - Fuzzy Jaccard similarity method for company name matching
- **`code-jaro_refactored.py`** - Jaro-Winkler similarity method for company name matching
- **`code-location-2_refactored.py`** - Location-based deduplication using geographic coordinates
- **`code-metaphone_refactored.py`** - Metaphone phonetic similarity method for company name matching

### Output Structure
```
individual_output/
└── new/
    ├── analysis_files/     # Detailed analysis reports
    ├── csv_files/
    │   ├── canonical/      # Canonical company representations
    │   └── clustered/      # Companies with cluster assignments
```

## Shared Deduplication Utilities

The `shared_deduplication_utils.py` module provides common functionality:

### Data Loading and Preparation
- `load_and_prepare_data()` - Loads CSV data and cleans company names
- Removes punctuation, converts to lowercase, strips whitespace

### Similarity Functions
- `fuzzy_jaccard_similarity()` - Token-based fuzzy matching
- `jaro_winkler_similarity()` - String edit distance with prefix bonus
- `metaphone_similarity()` - Phonetic code comparison
- `location_similarity()` - Geographic coordinate proximity

### Clustering and Analysis
- `calculate_similarity_matrix()` - Computes pairwise similarities
- `perform_clustering()` - DBSCAN clustering on similarity matrix
- `create_cluster_dataframe()` - Adds cluster assignments to data
- `create_canonical_companies()` - Generates representative companies
- `save_results()` - Saves outputs to structured folders

## Deduplication Methods

### 1. Fuzzy Jaccard (`code-fuzzy_jaccard_refactored.py`)
- **Purpose**: Token-based similarity with fuzzy matching
- **Process**: Tokenizes company names, calculates fuzzy intersection/union
- **Threshold**: 0.8 (80% similarity)
- **Use Case**: Handles word order variations and typos

### 2. Jaro-Winkler (`code-jaro_refactored.py`)
- **Purpose**: String edit distance with prefix weighting
- **Process**: Character-level similarity with bonus for matching prefixes
- **Threshold**: 0.85 (85% similarity)
- **Use Case**: Handles spelling variations and abbreviations

### 3. Metaphone (`code-metaphone_refactored.py`)
- **Purpose**: Phonetic similarity matching
- **Process**: Converts names to phonetic codes, compares codes
- **Threshold**: 0.8 (80% similarity)
- **Use Case**: Handles pronunciation variations and spelling differences

### 4. Location-Based (`code-location-2_refactored.py`)
- **Purpose**: Geographic clustering of companies
- **Process**: Groups companies by coordinate proximity
- **Threshold**: 0.01 degrees (approximately 1.1 km)
- **Use Case**: Identifies companies at same/similar locations

### 5. Ensemble Location (`code-ensemble_loc.py`)
- **Purpose**: Combines name and location similarity
- **Process**: Weighted combination of name (40%) and location (60%) similarity
- **Threshold**: 0.7 (70% combined similarity)
- **Use Case**: Most robust method, handles both name and location variations

## Output File Format

### Clustered Companies CSV
- All original columns plus `cluster_id`
- `cluster_id = -1` indicates unclustered companies (noise)

### Canonical Companies CSV
- `canonical_company_id`: Unique identifier for canonical company
- `canonical_company_name`: Representative company name
- `cluster_id`: Associated cluster identifier
- `cluster_size`: Number of companies in cluster
- `member_company_ids`: Comma-separated list of member IDs
- `member_company_names`: Semicolon-separated list of member names

### Analysis Files
- Cluster statistics and size distribution
- Detailed list of companies in each cluster
- Performance metrics and thresholds used

## Old vs New Outputs

### Old Output Structure
- Files saved in root directory
- Mixed naming conventions
- No organized folder structure
- Large files causing git issues

### New Output Structure
- Organized in `individual_output/new/`
- Consistent naming: `{method}_{type}.csv/txt`
- Separate folders for analysis, canonical, and clustered files
- Large files properly ignored in `.gitignore`

## Ensemble Location Approach

The ensemble method combines multiple similarity metrics:

1. **Name Similarity** (40% weight)
   - Uses rapidfuzz ratio for string similarity
   - Handles spelling variations and typos

2. **Location Similarity** (60% weight)
   - Geographic proximity calculation
   - Accounts for coordinate precision

3. **Combined Score**
   - Weighted average of name and location similarity
   - Threshold of 0.7 for clustering

## Manual Approval Process

For high-confidence clusters, manual review is recommended:

1. **Review Large Clusters** (>5 companies)
2. **Check Edge Cases** (similar names, different locations)
3. **Validate Canonical Representatives**
4. **Adjust Thresholds** if needed

## Rejected Files

The following files were removed from the repository:
- Large HTML files (>100MB)
- Old unrefactored scripts
- Duplicate analysis files
- Temporary output files

## ML Training Plans

Future enhancements could include:

1. **Supervised Learning**
   - Train on manually labeled similar/dissimilar pairs
   - Use features from multiple similarity methods
   - Implement gradient boosting or neural networks

2. **Active Learning**
   - Identify uncertain cases for manual review
   - Iteratively improve model with feedback
   - Reduce manual labeling effort

3. **Feature Engineering**
   - Extract business entity types
   - Use industry classification codes
   - Incorporate temporal information

## Usage Instructions

### Prerequisites
```bash
pip install pandas numpy scikit-learn rapidfuzz tqdm metaphone
```

### Running Individual Methods
```bash
python code-fuzzy_jaccard_refactored.py
python code-jaro_refactored.py
python code-metaphone_refactored.py
python code-location-2_refactored.py
python code-ensemble_loc.py
```

### Input Data Format
CSV file with columns:
- `company_id`: Unique identifier
- `company_name`: Company name
- `latitude`: Geographic latitude
- `longitude`: Geographic longitude
- Additional metadata columns as needed

### Output Location
All results are saved to `individual_output/new/` with organized subfolders for easy analysis and comparison.

## Performance Considerations

- **Memory Usage**: Similarity matrices can be large for big datasets
- **Processing Time**: O(n²) complexity for pairwise comparisons
- **Scalability**: Consider blocking strategies for large datasets
- **Accuracy**: Ensemble method provides best balance of precision/recall

## Future Improvements

1. **Parallel Processing**: Implement multiprocessing for similarity calculations
2. **Blocking Strategies**: Reduce comparisons using indexing
3. **Incremental Updates**: Handle new data without full reprocessing
4. **API Integration**: Real-time deduplication service
5. **Visualization**: Interactive cluster exploration tools 