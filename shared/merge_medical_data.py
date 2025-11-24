import pandas as pd
import numpy as np

def merge_medical_csvs():
    """
    Merge discharge.csv, diagnoses_icd.csv, and d_icd_diagnoses.csv into a single comprehensive CSV
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
        # This links icd_code with its long_title description
        print("Step 1: Merging diagnoses with ICD descriptions...")
        diagnoses_with_descriptions = pd.merge(
            diagnoses_df, 
            icd_descriptions_df, 
            on=['icd_code', 'icd_version'], 
            how='left'
        )
        print(f"After merging diagnoses with descriptions: {len(diagnoses_with_descriptions)} rows")
        
        # Step 2: Merge with discharge notes
        # This links subject_id and hadm_id with discharge text
        print("Step 2: Merging with discharge notes...")
        final_merged = pd.merge(
            diagnoses_with_descriptions,
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
        
        # Handle missing values - keep them as they are (NaN will be preserved)
        print(f"\nFinal dataset shape: {final_df.shape}")
        print(f"Columns: {list(final_df.columns)}")
        
        # Check for missing values
        print("\nMissing values summary:")
        missing_summary = final_df.isnull().sum()
        for col, missing_count in missing_summary.items():
            if missing_count > 0:
                print(f"  {col}: {missing_count} missing values")
        
        # Save to new CSV
        output_filename = 'merged_medical_data.csv'
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
                else:
                    print(f"  {col}: {value}")
            print("-" * 50)
        
        print(f"\n✅ Successfully created merged dataset with {len(final_df)} rows")
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
    print(f"Unique ICD codes: {df['icd_code'].nunique()}")
    
    # Check for subjects with multiple admissions
    subjects_multi_admissions = df.groupby('subject_id')['hadm_id'].nunique()
    multi_admission_subjects = subjects_multi_admissions[subjects_multi_admissions > 1]
    print(f"Subjects with multiple admissions: {len(multi_admission_subjects)}")
    
    # Check for admissions with multiple diagnoses
    admissions_multi_diagnoses = df.groupby('hadm_id')['icd_code'].nunique()
    multi_diagnosis_admissions = admissions_multi_diagnoses[admissions_multi_diagnoses > 1]
    print(f"Admissions with multiple diagnoses: {len(multi_diagnosis_admissions)}")

if __name__ == "__main__":
    print("Medical CSV Merger")
    print("=" * 50)
    print("This script merges discharge.csv, diagnoses_icd.csv, and d_icd_diagnoses.csv")
    print("into a single comprehensive dataset.\n")
    
    # Perform the merge
    merged_df = merge_medical_csvs()
    
    # Validate results if merge was successful
    if merged_df is not None:
        validate_merge_results(merged_df)
        
        print("\n" + "="*60)
        print("✅ MERGE COMPLETED SUCCESSFULLY!")
        print("✅ Output file: merged_medical_data.csv")
        print("="*60)
