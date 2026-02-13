"""
Unified Preprocessing Entry Point for EEGDM Integration
This module provides the entry point for the new unified preprocessing pipeline
that can be called from EEGDM's main.py workflow.
"""

import os
import shutil
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from .process_edf_to_raw_pkl import UnifiedEDFProcessor


def entry(config: DictConfig):
    """
    Entry point for unified preprocessing - compatible with EEGDM main.py
    
    Args:
        config: Hydra configuration containing preprocessing settings
    """
    print("Starting unified preprocessing...")
    
    # Extract configuration
    input_config = config.get("input_files", {})
    output_config = config.get("output_files", {})
    preprocessing_config = config.get("preprocessing", {})
    
    # Initialize processor
    processor = UnifiedEDFProcessor(
        sampling_rate=preprocessing_config.get("sampling_rate", 256),
        segment_length=preprocessing_config.get("segment_length", 10)
    )
    
    # Create output directories
    output_root = output_config["root"]
    train_dir = os.path.join(output_root, output_config["train_dir"])
    val_dir = os.path.join(output_root, output_config["val_dir"])
    test_dir = os.path.join(output_root, output_config["test_dir"])
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    data_type = preprocessing_config.get("data_type", "auto")
    
    if data_type == "chbmit":
        process_chbmit_data(config, processor, train_dir, val_dir, test_dir)
    elif data_type == "palantir":
        process_palantir_data(config, processor, train_dir, val_dir, test_dir)
    else:
        # Auto-detect based on input path
        input_root = input_config["root"]
        if "chb" in input_root.lower():
            data_type = "chbmit"
            process_chbmit_data(config, processor, train_dir, val_dir, test_dir)
        elif "palantir" in input_root.lower():
            data_type = "palantir"
            process_palantir_data(config, processor, train_dir, val_dir, test_dir)
        else:
            raise ValueError(f"Cannot auto-detect data type from path: {input_root}")
    
    print(f"Unified preprocessing complete for {data_type} data!")
    print(f"Output saved to: {output_root}")


def process_chbmit_data(config: DictConfig, processor: UnifiedEDFProcessor, 
                       train_dir: str, val_dir: str, test_dir: str):
    """Process CHB-MIT data with predefined patient splits"""
    
    input_config = config["input_files"]
    preprocessing_config = config["preprocessing"]
    
    # Get patient splits from config
    train_patients = preprocessing_config.get("train_patients", [])
    val_patients = preprocessing_config.get("val_patients", [])
    test_patients = preprocessing_config.get("test_patients", [])
    
    input_root = input_config["root"]
    
    # Check if we're processing from existing processed data or raw EDF files
    if os.path.exists(os.path.join(input_root, "test")) and os.path.exists(os.path.join(input_root, "train")):
        # We have existing processed structure - reprocess it
        print("Found existing processed structure, reprocessing...")
        
        # Process test data (existing structure)
        existing_test_dir = os.path.join(input_root, "test")
        if os.path.exists(existing_test_dir):
            edf_files = [f for f in os.listdir(existing_test_dir) if f.lower().endswith('.edf')]
            print(f"Processing {len(edf_files)} test EDF files...")
            
            for edf_file in edf_files:
                edf_path = os.path.join(existing_test_dir, edf_file)
                # Look for summary file
                summary_path = None
                summary_files = [f for f in os.listdir(existing_test_dir) if f.endswith('-summary.txt')]
                if summary_files:
                    # Match patient ID
                    patient_id = edf_file.split('_')[0]  # e.g., chb01 from chb01_01.edf
                    for summary_file in summary_files:
                        if patient_id in summary_file:
                            summary_path = os.path.join(existing_test_dir, summary_file)
                            break
                
                processor.process_edf_file(edf_path, test_dir, summary_path)
                
    else:
        # Process from raw patient directories
        print("Processing from raw patient directories...")
        
        # Find patient directories
        patient_dirs = []
        if os.path.exists(input_root):
            for item in os.listdir(input_root):
                item_path = os.path.join(input_root, item)
                if os.path.isdir(item_path) and item.startswith('chb'):
                    patient_dirs.append(item)
        
        print(f"Found patient directories: {patient_dirs}")
        
        # Process each patient based on split
        for patient_id in patient_dirs:
            patient_path = os.path.join(input_root, patient_id)
            
            if patient_id in test_patients:
                output_dir = test_dir
                split = "test"
            elif patient_id in val_patients:
                output_dir = val_dir
                split = "val"
            else:
                output_dir = train_dir
                split = "train"
                
            print(f"Processing {patient_id} -> {split}")
            processor.process_chbmit_patient(patient_path, output_dir)


def process_palantir_data(config: DictConfig, processor: UnifiedEDFProcessor,
                         train_dir: str, val_dir: str, test_dir: str):
    """Process Palantir data with random splits"""
    
    input_config = config["input_files"] 
    preprocessing_config = config["preprocessing"]
    
    input_root = input_config["root"]
    
    # Get split ratios
    train_ratio = preprocessing_config.get("train_ratio", 0.7)
    val_ratio = preprocessing_config.get("val_ratio", 0.15)
    test_ratio = preprocessing_config.get("test_ratio", 0.15)
    random_seed = preprocessing_config.get("random_seed", 42)
    
    print(f"Processing Palantir data with splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Process all files to a temporary directory first
    temp_dir = os.path.join(os.path.dirname(train_dir), "temp_processed")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Process all Palantir files
        all_files = processor.process_palantir_files(input_root, temp_dir)
        
        if not all_files:
            print("No files were processed!")
            return
            
        # Split files randomly
        np.random.seed(random_seed)
        np.random.shuffle(all_files)
        
        n_total = len(all_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        print(f"Splitting {n_total} files: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
        # Move files to appropriate directories
        for file_list, target_dir in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
            for file_path in file_list:
                filename = os.path.basename(file_path)
                target_path = os.path.join(target_dir, filename)
                shutil.move(file_path, target_path)
                
    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# Backwards compatibility alias
load_files = entry
