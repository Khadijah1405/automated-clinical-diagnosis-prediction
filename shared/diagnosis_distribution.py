import pandas as pd
import numpy as np
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_icd_distribution(csv_file):
    """
    Analyze ICD code and long title distribution in the dataset.
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    
    # Clean and prepare data
    df_clean = df.dropna(subset=['icd_code', 'long_title'])
    print(f"Samples with valid ICD codes and titles: {len(df_clean)}")
    
    return analyze_icd_codes(df_clean), analyze_long_titles(df_clean), analyze_icd_versions(df_clean)

def analyze_icd_codes(df):
    """
    Analyze ICD code distribution.
    """
    print(f"\n" + "="*50)
    print(f"ICD CODE ANALYSIS")
    print(f"="*50)
    
    icd_codes = df['icd_code'].dropna()
    icd_counts = Counter(icd_codes)
    unique_codes = len(icd_counts)
    
    print(f"Total ICD code entries: {len(icd_codes)}")
    print(f"Unique ICD codes: {unique_codes}")
    print(f"Average occurrences per ICD code: {len(icd_codes) / unique_codes:.2f}")
    
    # Show most common ICD codes
    print(f"\nTop 20 most frequent ICD codes:")
    for i, (code, count) in enumerate(icd_counts.most_common(20)):
        percentage = (count / len(icd_codes)) * 100
        # Get corresponding long title for this code
        sample_title = df[df['icd_code'] == code]['long_title'].iloc[0]
        truncated_title = sample_title[:60] + "..." if len(sample_title) > 60 else sample_title
        print(f"{i+1:2d}. {code:8s} ({count:4d} | {percentage:5.2f}%) - {truncated_title}")
    
    # Analyze frequency distribution
    counts = list(icd_counts.values())
    print(f"\n=== ICD CODE FREQUENCY DISTRIBUTION ===")
    print(f"Codes appearing once: {sum(1 for c in counts if c == 1)} ({sum(1 for c in counts if c == 1)/unique_codes*100:.1f}%)")
    print(f"Codes appearing 2-5 times: {sum(1 for c in counts if 2 <= c <= 5)} ({sum(1 for c in counts if 2 <= c <= 5)/unique_codes*100:.1f}%)")
    print(f"Codes appearing 6-20 times: {sum(1 for c in counts if 6 <= c <= 20)} ({sum(1 for c in counts if 6 <= c <= 20)/unique_codes*100:.1f}%)")
    print(f"Codes appearing >20 times: {sum(1 for c in counts if c > 20)} ({sum(1 for c in counts if c > 20)/unique_codes*100:.1f}%)")
    
    # Analyze ICD code patterns
    print(f"\n=== ICD CODE PATTERNS ===")
    # Extract main categories (first character for ICD-10, first 3 digits for ICD-9)
    icd_9_codes = [code for code in icd_codes if code.replace('.', '').isdigit()]
    icd_10_codes = [code for code in icd_codes if not code.replace('.', '').isdigit()]
    
    print(f"ICD-9 codes: {len(icd_9_codes)} ({len(icd_9_codes)/len(icd_codes)*100:.1f}%)")
    print(f"ICD-10 codes: {len(icd_10_codes)} ({len(icd_10_codes)/len(icd_codes)*100:.1f}%)")
    
    # Analyze ICD-10 categories (first letter)
    if icd_10_codes:
        icd_10_categories = Counter([code[0] for code in icd_10_codes if len(code) > 0])
        print(f"\nTop ICD-10 categories (by first letter):")
        for category, count in icd_10_categories.most_common(10):
            percentage = (count / len(icd_10_codes)) * 100
            print(f"  {category}: {count:4d} ({percentage:5.2f}%)")
    
    return icd_counts

def analyze_long_titles(df):
    """
    Analyze long title distribution.
    """
    print(f"\n" + "="*50)
    print(f"LONG TITLE ANALYSIS")
    print(f"="*50)
    
    long_titles = df['long_title'].dropna()
    title_counts = Counter(long_titles)
    unique_titles = len(title_counts)
    
    print(f"Total long title entries: {len(long_titles)}")
    print(f"Unique long titles: {unique_titles}")
    print(f"Average occurrences per title: {len(long_titles) / unique_titles:.2f}")
    
    # Show most common long titles
    print(f"\nTop 20 most frequent long titles:")
    for i, (title, count) in enumerate(title_counts.most_common(20)):
        percentage = (count / len(long_titles)) * 100
        # Get corresponding ICD code for this title
        sample_code = df[df['long_title'] == title]['icd_code'].iloc[0]
        truncated_title = title[:70] + "..." if len(title) > 70 else title
        print(f"{i+1:2d}. ({sample_code:8s}) ({count:4d} | {percentage:5.2f}%) {truncated_title}")
    
    # Analyze title frequency distribution
    counts = list(title_counts.values())
    print(f"\n=== LONG TITLE FREQUENCY DISTRIBUTION ===")
    print(f"Titles appearing once: {sum(1 for c in counts if c == 1)} ({sum(1 for c in counts if c == 1)/unique_titles*100:.1f}%)")
    print(f"Titles appearing 2-5 times: {sum(1 for c in counts if 2 <= c <= 5)} ({sum(1 for c in counts if 2 <= c <= 5)/unique_titles*100:.1f}%)")
    print(f"Titles appearing 6-20 times: {sum(1 for c in counts if 6 <= c <= 20)} ({sum(1 for c in counts if 6 <= c <= 20)/unique_titles*100:.1f}%)")
    print(f"Titles appearing >20 times: {sum(1 for c in counts if c > 20)} ({sum(1 for c in counts if c > 20)/unique_titles*100:.1f}%)")
    
    # Analyze title lengths
    title_lengths = [len(title) for title in long_titles]
    print(f"\n=== TITLE LENGTH STATISTICS ===")
    print(f"Average title length: {np.mean(title_lengths):.1f} characters")
    print(f"Median title length: {np.median(title_lengths):.1f} characters")
    print(f"Shortest title: {min(title_lengths)} characters")
    print(f"Longest title: {max(title_lengths)} characters")
    
    # Show some examples of short and long titles
    sorted_by_length = sorted(title_counts.items(), key=lambda x: len(x[0]))
    print(f"\nShortest titles:")
    for title, count in sorted_by_length[:5]:
        print(f"  ({count:2d}x) {title}")
    
    print(f"\nLongest titles:")
    for title, count in sorted(title_counts.items(), key=lambda x: len(x[0]), reverse=True)[:5]:
        print(f"  ({count:2d}x) {title[:100]}...")
    
    return title_counts

def analyze_icd_versions(df):
    """
    Analyze ICD version distribution.
    """
    print(f"\n" + "="*50)
    print(f"ICD VERSION ANALYSIS")
    print(f"="*50)
    
    if 'icd_version' in df.columns:
        version_counts = Counter(df['icd_version'].dropna())
        print(f"ICD version distribution:")
        for version, count in version_counts.most_common():
            percentage = (count / len(df)) * 100
            print(f"  ICD-{version}: {count:4d} ({percentage:5.2f}%)")
    else:
        print("No ICD version column found in dataset")
    
    return version_counts if 'icd_version' in df.columns else None

def analyze_code_title_consistency(df):
    """
    Check for consistency between ICD codes and long titles.
    """
    print(f"\n" + "="*50)
    print(f"CODE-TITLE CONSISTENCY ANALYSIS")
    print(f"="*50)
    
    # Group by ICD code and check if titles are consistent
    code_title_mapping = df.groupby('icd_code')['long_title'].unique()
    
    inconsistent_codes = []
    for code, titles in code_title_mapping.items():
        if len(titles) > 1:
            inconsistent_codes.append((code, titles))
    
    print(f"Total unique ICD codes: {len(code_title_mapping)}")
    print(f"Codes with multiple titles: {len(inconsistent_codes)}")
    
    if inconsistent_codes:
        print(f"\nTop 10 codes with multiple titles:")
        for i, (code, titles) in enumerate(inconsistent_codes[:10]):
            print(f"\n{i+1}. Code {code} has {len(titles)} different titles:")
            for j, title in enumerate(titles[:3]):  # Show first 3 titles
                truncated = title[:80] + "..." if len(title) > 80 else title
                print(f"   {j+1}. {truncated}")
            if len(titles) > 3:
                print(f"   ... and {len(titles)-3} more")
    
    # Group by title and check if codes are consistent
    title_code_mapping = df.groupby('long_title')['icd_code'].unique()
    inconsistent_titles = []
    for title, codes in title_code_mapping.items():
        if len(codes) > 1:
            inconsistent_titles.append((title, codes))
    
    print(f"\nTitles with multiple codes: {len(inconsistent_titles)}")
    if inconsistent_titles:
        print(f"Top 5 titles with multiple codes:")
        for i, (title, codes) in enumerate(inconsistent_titles[:5]):
            truncated_title = title[:60] + "..." if len(title) > 60 else title
            codes_str = ", ".join(codes[:5])
            if len(codes) > 5:
                codes_str += f" ... and {len(codes)-5} more"
            print(f"  {i+1}. {truncated_title}")
            print(f"     Codes: {codes_str}")

def create_visualizations(icd_counts, title_counts):
    """
    Create visualizations for ICD code and title distributions.
    """
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top ICD codes
        top_codes = dict(list(icd_counts.most_common(15)))
        axes[0,0].barh(list(top_codes.keys()), list(top_codes.values()))
        axes[0,0].set_title('Top 15 Most Frequent ICD Codes')
        axes[0,0].set_xlabel('Frequency')
        
        # ICD code frequency distribution
        code_freq_dist = Counter(icd_counts.values())
        axes[0,1].bar(list(code_freq_dist.keys())[:20], list(code_freq_dist.values())[:20])
        axes[0,1].set_title('ICD Code Frequency Distribution')
        axes[0,1].set_xlabel('Times Code Appears')
        axes[0,1].set_ylabel('Number of Codes')
        
        # Top long titles (truncated for display)
        top_titles = dict(list(title_counts.most_common(10)))
        truncated_titles = [title[:30] + "..." if len(title) > 30 else title for title in top_titles.keys()]
        axes[1,0].barh(truncated_titles, list(top_titles.values()))
        axes[1,0].set_title('Top 10 Most Frequent Long Titles')
        axes[1,0].set_xlabel('Frequency')
        
        # Title frequency distribution
        title_freq_dist = Counter(title_counts.values())
        axes[1,1].bar(list(title_freq_dist.keys())[:20], list(title_freq_dist.values())[:20])
        axes[1,1].set_title('Long Title Frequency Distribution')
        axes[1,1].set_xlabel('Times Title Appears')
        axes[1,1].set_ylabel('Number of Titles')
        
        plt.tight_layout()
        plt.savefig('icd_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Visualization creation failed: {e}")

# Usage
if __name__ == "__main__":
    # Update this path to your dataset
    csv_file = "text_seperated_with_icd.csv"
    
    try:
        # Main analysis
        icd_counts, title_counts, version_counts = analyze_icd_distribution(csv_file)
        
        # Consistency analysis
        df = pd.read_csv(csv_file)
        analyze_code_title_consistency(df)
        
        # Create visualizations
        create_visualizations(icd_counts, title_counts)
        
        print(f"\n" + "="*50)
        print(f"ANALYSIS COMPLETE")
        print(f"="*50)
        print(f"✅ ICD code analysis completed")
        print(f"✅ Long title analysis completed")
        print(f"✅ Version analysis completed")
        print(f"✅ Consistency check completed")
        print(f"✅ Visualizations saved as 'icd_analysis_plots.png'")
        
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        print("Please update the csv_file path to your actual dataset location")
    except Exception as e:
        print(f"Analysis failed: {e}")
        print("Please check your data format and file path")
