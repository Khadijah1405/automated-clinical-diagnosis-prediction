import pandas as pd

def count_unique_hcup_categories(csv_file_path):
    """
    Count unique samples in the hcup_primary_category column from a CSV file.
    
    Args:
        csv_file_path (str): Path to the CSV file
    
    Returns:
        dict: Dictionary with counts and unique values
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Check if the column exists
        if 'hcup_primary_category' not in df.columns:
            print("Error: Column 'hcup_primary_category' not found in the CSV file.")
            print("Available columns:", list(df.columns))
            return None
        
        # Get the hcup_primary_category column
        hcup_column = df['hcup_primary_category']
        
        # Count unique values (excluding NaN/null values)
        unique_values = hcup_column.dropna().unique()
        unique_count = len(unique_values)
        
        # Get value counts for each unique category
        value_counts = hcup_column.value_counts(dropna=False)
        
        # Print results
        print(f"Dataset: {csv_file_path}")
        print(f"Total rows in dataset: {len(df)}")
        print(f"Total unique categories in 'hcup_primary_category': {unique_count}")
        print(f"Null/NaN values: {hcup_column.isnull().sum()}")
        print("\n" + "="*50)
        print("Unique categories:")
        print("="*50)
        
        for i, category in enumerate(sorted(unique_values), 1):
            print(f"{i:2d}. {category}")
        
        print("\n" + "="*50)
        print("Category counts (including null values):")
        print("="*50)
        print(value_counts)
        
        # Return summary dictionary
        return {
            'total_rows': len(df),
            'unique_count': unique_count,
            'null_count': hcup_column.isnull().sum(),
            'unique_values': sorted(unique_values.tolist()),
            'value_counts': value_counts.to_dict()
        }
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    # Specify your CSV file path
    csv_file = "textfinal_hcup_categories_FIXED.csv"
    
    # Count unique samples
    results = count_unique_hcup_categories(csv_file)
    
    # Optional: Save results to a text file
    if results:
        with open("hcup_category_analysis.txt", "w") as f:
            f.write(f"Analysis of hcup_primary_category column\n")
            f.write("="*50 + "\n")
            f.write(f"Total rows: {results['total_rows']}\n")
            f.write(f"Unique categories: {results['unique_count']}\n")
            f.write(f"Null values: {results['null_count']}\n\n")
            
            f.write("Unique category list:\n")
            for i, cat in enumerate(results['unique_values'], 1):
                f.write(f"{i:2d}. {cat}\n")
            
            f.write(f"\nValue counts:\n")
            for category, count in results['value_counts'].items():
                f.write(f"{category}: {count}\n")
        
        print(f"\nResults saved to 'hcup_category_analysis.txt'")
