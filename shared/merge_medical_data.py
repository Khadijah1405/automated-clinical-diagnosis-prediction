import pandas as pd
import numpy as np

def merge_medical_csvs():
    """
    Merge discharge.csv, diagnoses_icd.csv, and d_icd_diagnoses.csv into a single comprehensive CSV
    with aggregated diagnosis information to avoid row duplication
    """
    
    try:
        # Read the CSV files
        print("Reading CSV files...")
        
        # Read discharge notes
        discharge_df = pd.read_csv('discharge.csv')
        print(f"Loaded discharge.csv: {len(discharge_df)} rows")
        
        # Read diagnoses with ICD codes
        diagnoses_df = pd.read_csv('diagnoses_icd.csv')
        print(f"Loaded diagnoses_icd.csv: {len(diagnoses_df)} rows")
        
        # Read ICD code descriptions
        icd_descriptions_df = pd.read_csv('d_icd_diagnoses.csv')
        print(f"Loaded d_icd_diagnoses.csv: {len(icd_descriptions_df)} rows")
        
        # Clean column names (remove any extra spaces)
        discharge_df.columns = discharge_df.columns.str.strip()
        diagnoses_df.columns = diagnoses_df.columns.str.strip()
        icd_descriptions_df.columns = icd_descriptions_df.columns.str.strip()
        
        print("\nStarting merge process...")
        
        # Step 1: Merge diagnoses with ICD descriptions
        print("Step 1: Merging diagnoses with ICD descriptions...")
        diagnoses_with_descriptions = pd.merge(
            diagnoses_df, 
            icd_descriptions_df, 
            on=['icd_code', 'icd_version'], 
            how='left'
        )
        print(f"After merging diagnoses with descriptions: {len(diagnoses_with_descriptions)} rows")
        
        # Step 2: Aggregate diagnosis information by subject_id and hadm_id
        print("Step 2: Aggregating diagnosis information...")
        
        # Function to safely join non-null values
        def safe_join(series, separator=', '):
            # Remove NaN values and convert to string
            clean_values = series.dropna().astype(str)
            if len(clean_values) == 0:
                return np.nan
            return separator.join(clean_values)
        
        # Group by subject_id and hadm_id, then aggregate the diagnosis columns
        aggregated_diagnoses = diagnoses_with_descriptions.groupby(['subject_id', 'hadm_id']).agg({
            'seq_num': lambda x: safe_join(x, ', '),
            'icd_code': lambda x: safe_join(x, ', '),
            'icd_version': lambda x: safe_join(x, ', '),
            'long_title': lambda x: safe_join(x, ' | ')  # Using | separator for better readability of long titles
        }).reset_index()
        
        print(f"After aggregating diagnoses: {len(aggregated_diagnoses)} rows")
        
        # Step 3: Merge with discharge notes
        print("Step 3: Merging with discharge notes...")
        final_merged = pd.merge(
            aggregated_diagnoses,
            discharge_df[['subject_id', 'hadm_id', 'text']], 
            on=['subject_id', 'hadm_id'], 
            how='inner'
        )
        print(f"After merging with discharge notes: {len(final_merged)} rows")
        
        # Select and reorder columns as requested
        desired_columns = ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version', 'long_title', 'text']
        
        # Check if all desired columns exist
        missing_columns = [col for col in desired_columns if col not in final_merged.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        # Select available columns
        available_columns = [col for col in desired_columns if col in final_merged.columns]
        final_df = final_merged[available_columns].copy()
        
        print(f"\nFinal dataset shape: {final_df.shape}")
        print(f"Columns: {list(final_df.columns)}")
        
        # Check for missing values
        print("\nMissing values summary:")
        missing_summary = final_df.isnull().sum()
        for col, missing_count in missing_summary.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count} missing values")
        
        # Save to new CSV
        output_filename = 'merged_medical_data_with_comma.csv'
        final_df.to_csv(output_filename, index=False)
        print(f"\nMerged data saved to: {output_filename}")
        
        # Display sample of the merged data
        print("\nSample of merged data (first 3 rows):")
        print("="*100)
        for i, row in final_df.head(3).iterrows():
            print(f"Row {i+1}:")
            for col in final_df.columns:
                value = row[col]
                if pd.isna(value):
                    print(f"  {col}: [MISSING]")
                elif col == 'text' and len(str(value)) > 100:
                    print(f"  {col}: {str(value)[:100]}...")
                elif col in ['seq_num', 'icd_code', 'icd_version', 'long_title'] and len(str(value)) > 150:
                    print(f"  {col}: {str(value)[:150]}...")
                else:
                    print(f"  {col}: {value}")
            print("-" * 50)
        
        print(f"\n✅ Successfully created merged dataset with {len(final_df)} rows")
        print("✅ Diagnosis information has been aggregated to avoid row duplication")
        return final_df
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
        print("Please ensure all three CSV files are in the same directory as this script:")
        print("  - discharge.csv")
        print("  - diagnoses_icd.csv") 
        print("  - d_icd_diagnoses.csv")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

def validate_merge_results(df):
    """
    Validate the merged results and provide statistics
    """
    print("\n" + "="*60)
    print("MERGE VALIDATION REPORT")
    print("="*60)
    
    print(f"Total records: {len(df)}")
    print(f"Unique subjects: {df['subject_id'].nunique()}")
    print(f"Unique admissions: {df['hadm_id'].nunique()}")
    
    # Count total ICD codes across all rows
    total_icd_codes = 0
    for idx, row in df.iterrows():
        if pd.notna(row['icd_code']):
            icd_codes = str(row['icd_code']).split(', ')
            total_icd_codes += len(icd_codes)
    
    print(f"Total ICD codes across all records: {total_icd_codes}")
    
    # Check for subjects with multiple admissions
    subjects_multi_admissions = df.groupby('subject_id')['hadm_id'].nunique()
    multi_admission_subjects = subjects_multi_admissions[subjects_multi_admissions > 1]
    print(f"Subjects with multiple admissions: {len(multi_admission_subjects)}")
    
    # Show statistics about diagnosis aggregation
    print("\nDiagnosis aggregation statistics:")
    
    # Count how many admissions have multiple diagnoses
    multi_diagnosis_count = 0
    max_diagnoses = 0
    
    for idx, row in df.iterrows():
        if pd.notna(row['icd_code']):
            icd_count = len(str(row['icd_code']).split(', '))
            if icd_count > 1:
                multi_diagnosis_count += 1
            max_diagnoses = max(max_diagnoses, icd_count)
    
    print(f"Admissions with multiple diagnoses: {multi_diagnosis_count}")
    print(f"Maximum diagnoses per admission: {max_diagnoses}")

def analyze_diagnosis_distribution(df):
    """
    Analyze how diagnoses are distributed across admissions
    """
    print("\n" + "="*60)
    print("DIAGNOSIS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    diagnosis_counts = []
    
    for idx, row in df.iterrows():
        if pd.notna(row['icd_code']):
            icd_count = len(str(row['icd_code']).split(', '))
            diagnosis_counts.append(icd_count)
        else:
            diagnosis_counts.append(0)
    
    diagnosis_counts = pd.Series(diagnosis_counts)
    
    print("Distribution of number of diagnoses per admission:")
    print(diagnosis_counts.value_counts().sort_index())
    
    print(f"\nAverage diagnoses per admission: {diagnosis_counts.mean():.2f}")
    print(f"Median diagnoses per admission: {diagnosis_counts.median():.2f}")

if __name__ == "__main__":
    print("Medical CSV Merger (with Diagnosis Aggregation)")
    print("=" * 60)
    print("This script merges discharge.csv, diagnoses_icd.csv, and d_icd_diagnoses.csv")
    print("into a single comprehensive dataset with aggregated diagnosis information.")
    print("Multiple diagnoses per admission are combined into comma-separated values.\n")
    
    # Perform the merge
    merged_df = merge_medical_csvs()
    
    # Validate results if merge was successful
    if merged_df is not None:
        validate_merge_results(merged_df)
        analyze_diagnosis_distribution(merged_df)
        
        print("\n" + "="*60)
        print("✅ MERGE COMPLETED SUCCESSFULLY!")
        print("✅ Output file: merged_medical_data.csv")
        print("✅ No duplicate rows - diagnosis data aggregated!")
        print("="*60)
