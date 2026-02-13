# bulk_preprocess_chb01.py (run from EEGDM-main/src/extra/)
from eegdm_preprocessing import EEGDMPreprocessor  # Local import
import os

# Initialize processor
processor = EEGDMPreprocessor(style='chbmit')

# Adjust paths relative to src/extra directory
input_dir = "../../data/chbmit/processed_seg/test"  # Up 2 levels to EEGDM-main, then down
output_dir = "../../data/chbmit/processed_seg/test/pkl"

# Convert to absolute paths to be safe
input_dir = os.path.abspath(input_dir)
output_dir = os.path.abspath(output_dir)

print(f"Processing EDF files from: {input_dir}")
print(f"Saving pkl files to: {output_dir}")

# Process all chb01 files
output_files = processor.process_directory(input_dir, output_dir)
print(f"Successfully processed {len(output_files)} segments")
