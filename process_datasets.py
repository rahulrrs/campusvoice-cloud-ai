import pandas as pd
import numpy as np

def process_nyc311_data(input_file, output_file=None, max_rows=50000):
    """
    Process NYC 311 Service Requests data into your complaint format.
    
    Args:
        input_file: Path to NYC 311 CSV file
        output_file: Path to save processed data (optional)
        max_rows: Maximum rows to process (None for all)
    
    Returns:
        DataFrame with columns: text, category, source
    """
    
    print(f"Loading NYC 311 data from {input_file}...")
    
    # Read the data
    if max_rows:
        df = pd.read_csv(input_file, nrows=max_rows, low_memory=False)
    else:
        df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Loaded {len(df)} rows")
    
    # Category mapping
    category_mapping = {
        # Noise complaints
        'Noise - Residential': 'Noise',
        'Noise - Street/Sidewalk': 'Noise',
        'Noise - Commercial': 'Noise',
        'Noise - Vehicle': 'Noise',
        'Noise - Park': 'Noise',
        'Noise': 'Noise',
        
        # HVAC
        'HEAT/HOT WATER': 'HVAC & Temperature Control',
        'HEATING': 'HVAC & Temperature Control',
        'Heat/Hot Water': 'HVAC & Temperature Control',
        
        # Plumbing
        'Plumbing': 'Water & Plumbing',
        'Water System': 'Water & Plumbing',
        'PLUMBING': 'Water & Plumbing',
        'Water Leak': 'Water & Plumbing',
        
        # Building & Maintenance
        'PAINT/PLASTER': 'Building & Maintenance',
        'General Construction/Plumbing': 'Building & Maintenance',
        'Elevator': 'Building & Maintenance',
        'DOOR/WINDOW': 'Building & Maintenance',
        'FLOORING/STAIRS': 'Building & Maintenance',
        'General': 'Building & Maintenance',
        
        # Electrical
        'Street Light Condition': 'Electrical',
        'Electric': 'Electrical',
        'ELECTRIC': 'Electrical',
        
        # Environment
        'Damaged Tree': 'Environment',
        'Air Quality': 'Environment',
        'Water Quality': 'Environment',
        
        # Graffiti
        'Graffiti': 'Graffiti & Vandalism',
        'Vandalism': 'Graffiti & Vandalism',
        
        # Parking
        'Illegal Parking': 'Parking & Vehicles',
        'Blocked Driveway': 'Parking & Vehicles',
        'Derelict Vehicle': 'Parking & Vehicles',
        
        # Sanitation
        'Rodent': 'Cleanliness & Sanitation',
        'Unsanitary Condition': 'Cleanliness & Sanitation',
        'Dirty Conditions': 'Cleanliness & Sanitation',
        'Missed Collection': 'Cleanliness & Sanitation',
        
        # Smoking
        'Smoking': 'Smoking',
        
        # Animals
        'Animal Abuse': 'Animal & Pet Issues',
        'Animal in a Park': 'Animal & Pet Issues',
        
        # Lost property
        'Lost Property': 'Lost & Found',
    }
    
    # Process the data
    processed_rows = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"Processing row {idx}...")
        
        complaint_type = row.get('Complaint Type', '')
        descriptor = row.get('Descriptor', '')
        
        # Map to category
        category = category_mapping.get(complaint_type)
        
        if category:
            # Create text from complaint type and descriptor
            if pd.notna(descriptor) and descriptor:
                text = f"{complaint_type}: {descriptor}"
            else:
                text = complaint_type
            
            processed_rows.append({
                'text': text,
                'category': category,
                'source': 'nyc_311'
            })
    
    result_df = pd.DataFrame(processed_rows)
    
    print(f"\nProcessed {len(result_df)} complaints")
    print(f"\nCategory distribution:")
    print(result_df['category'].value_counts())
    
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
    
    return result_df


def process_consumer_complaints_data(input_file, output_file=None, max_rows=10000):
    """
    Process Consumer Complaints data.
    
    Note: This data is primarily financial complaints, so may not map
    well to apartment/housing complaints. Use with caution.
    """
    
    print(f"Loading Consumer Complaints data from {input_file}...")
    
    if max_rows:
        df = pd.read_csv(input_file, nrows=max_rows, low_memory=False)
    else:
        df = pd.read_csv(input_file, low_memory=False)
    
    print(f"Loaded {len(df)} rows")
    
    # Find text column
    text_cols = ['Consumer Complaint', 'Consumer complaint narrative', 'Complaint', 'complaint']
    text_col = None
    for col in text_cols:
        if col in df.columns:
            text_col = col
            break
    
    if not text_col:
        print("ERROR: Could not find text column")
        return pd.DataFrame()
    
    # Clean and process
    result_df = df[[text_col]].copy()
    result_df.columns = ['text']
    result_df = result_df.dropna(subset=['text'])
    
    # These are financial complaints, map generically
    result_df['category'] = 'Student Affairs / University Complaint'
    result_df['source'] = 'cfpb_complaints'
    
    print(f"\nProcessed {len(result_df)} complaints")
    
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
    
    return result_df


if __name__ == '__main__':
    print("NYC 311 Data Processor")
    print("=" * 80)
    print("\nUsage:")
    print("  python process_datasets.py")
    print("\nOr import and use:")
    print("  from process_datasets import process_nyc311_data")
    print("  df = process_nyc311_data('nyc_311.csv', 'processed_311.csv')")
