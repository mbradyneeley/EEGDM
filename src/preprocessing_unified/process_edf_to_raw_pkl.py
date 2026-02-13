"""
Unified EDF to Raw Pickle Preprocessing Script
Based on CBraMod process1.py and process2.py, adapted for:
1. CHB-MIT dataset with summary.txt files
2. Palantir EDF files
3. Custom EDF files

This script converts EDF files to raw pickle format without filtering/resampling
(those are applied on-the-fly during training via data_transform_chbmit_filt)
"""

import os
import pickle
import numpy as np
import mne
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import argparse
import re
from typing import Dict, List, Tuple, Optional, Union

# Suppress MNE verbose output
mne.set_log_level("CRITICAL")


class UnifiedEDFProcessor:
    """Unified EDF processor that can handle both CHB-MIT and Palantir data"""
    
    # Standard 16-channel bipolar montage used by EEGDM
    STANDARD_CHANNELS = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2", 
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
    ]
    
    # Channel name mappings for different naming conventions
    CHANNEL_ALIASES = {
        # 10-20 to 10-10 temporal mapping
        "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
        # Common variations
        "FP": "FP", "Fp": "FP",
        # Handle EEG prefixes
        "EEG FP1-REF": "FP1", "EEG FP2-REF": "FP2",
        "EEG F3-REF": "F3", "EEG F4-REF": "F4",
        "EEG C3-REF": "C3", "EEG C4-REF": "C4",
        "EEG P3-REF": "P3", "EEG P4-REF": "P4",
        "EEG O1-REF": "O1", "EEG O2-REF": "O2",
        "EEG F7-REF": "F7", "EEG F8-REF": "F8",
        "EEG T3-REF": "T7", "EEG T4-REF": "T8",
        "EEG T5-REF": "P7", "EEG T6-REF": "P8",
    }
    
    def __init__(self, sampling_rate: int = 256, segment_length: int = 10):
        """
        Initialize the processor
        
        Args:
            sampling_rate: Expected sampling rate of input data
            segment_length: Length of segments in seconds
        """
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.segment_samples = sampling_rate * segment_length
        
    def normalize_channel_name(self, channel_name: str) -> str:
        """Normalize channel names to standard format"""
        # Remove common prefixes/suffixes
        name = channel_name.upper().strip()
        name = re.sub(r'^EEG\s*', '', name)  # Remove EEG prefix
        name = re.sub(r'-REF$', '', name)   # Remove -REF suffix
        name = re.sub(r'-\d+$', '', name)   # Remove duplicate suffixes like -0, -1
        
        # Apply aliases
        for alias, standard in self.CHANNEL_ALIASES.items():
            name = name.replace(alias, standard)
            
        return name
        
    def create_bipolar_montage(self, raw: mne.io.Raw) -> Tuple[np.ndarray, List[str]]:
        """
        Create bipolar montage from monopolar channels
        
        Returns:
            signal: np.array of shape (n_channels, n_samples)
            channel_names: List of created bipolar channel names
        """
        # Normalize all channel names
        normalized_channels = {}
        for ch in raw.ch_names:
            normalized = self.normalize_channel_name(ch)
            normalized_channels[normalized] = ch
            
        print(f"Available normalized channels: {list(normalized_channels.keys())}")
        
        # Define bipolar pairs to create
        bipolar_pairs = [
            ('FP1', 'F7'), ('F7', 'T7'), ('T7', 'P7'), ('P7', 'O1'),
            ('FP2', 'F8'), ('F8', 'T8'), ('T8', 'P8'), ('P8', 'O2'),
            ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'), 
            ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')
        ]
        
        bipolar_signals = []
        bipolar_names = []
        
        for anode, cathode in bipolar_pairs:
            if anode in normalized_channels and cathode in normalized_channels:
                anode_orig = normalized_channels[anode]
                cathode_orig = normalized_channels[cathode]
                
                anode_data = raw.get_data(picks=[anode_orig])[0]
                cathode_data = raw.get_data(picks=[cathode_orig])[0]
                
                bipolar_signal = anode_data - cathode_data
                bipolar_signals.append(bipolar_signal)
                bipolar_names.append(f"{anode}-{cathode}")
                
                print(f"Created {anode}-{cathode} from {anode_orig} - {cathode_orig}")
            else:
                missing = []
                if anode not in normalized_channels:
                    missing.append(anode)
                if cathode not in normalized_channels:
                    missing.append(cathode)
                print(f"Cannot create {anode}-{cathode}: missing {missing}")
        
        if not bipolar_signals:
            raise ValueError("No bipolar channels could be created!")
            
        signal_array = np.array(bipolar_signals)
        print(f"Created bipolar montage with {len(bipolar_names)} channels: {bipolar_names}")
        
        return signal_array, bipolar_names
        
    def extract_bipolar_channels(self, raw: mne.io.Raw) -> Tuple[np.ndarray, List[str]]:
        """
        Extract existing bipolar channels or create them from monopolar
        
        Returns:
            signal: np.array of shape (n_channels, n_samples) 
            channel_names: List of channel names
        """
        # Check if data already has bipolar channels
        bipolar_count = sum(1 for ch in raw.ch_names if '-' in ch)
        
        if bipolar_count > len(raw.ch_names) / 2:
            print("Data appears to already be in bipolar format")
            # Extract existing bipolar channels
            bipolar_channels = [ch for ch in raw.ch_names if '-' in ch]
            
            # Try to match with standard channels, handling duplicates by taking first occurrence
            matched_channels = []
            used_standard_channels = set()
            
            for std_ch in self.STANDARD_CHANNELS:
                if std_ch in used_standard_channels:
                    continue
                    
                for bp_ch in bipolar_channels:
                    if self.channels_match(std_ch, bp_ch):
                        # Check if we already used this bipolar channel (handle duplicates)
                        if bp_ch not in [mc for mc, _ in matched_channels]:
                            matched_channels.append((bp_ch, std_ch))
                            used_standard_channels.add(std_ch)
                            print(f"Matched {bp_ch} -> {std_ch}")
                            break
                        else:
                            print(f"Skipping duplicate channel: {bp_ch}")
                            
            if matched_channels:
                # Extract only the original channel names for data extraction
                channel_names = [mc[0] for mc in matched_channels]
                print(f"Selected {len(channel_names)} bipolar channels (avoiding duplicates)")
                signal = raw.get_data(picks=channel_names)
                return signal, channel_names
                
        # Create bipolar montage from monopolar channels
        print("Creating bipolar montage from monopolar channels")
        return self.create_bipolar_montage(raw)
        
    def channels_match(self, standard: str, candidate: str) -> bool:
        """Check if a candidate channel matches a standard channel name"""
        # Remove MNE's automatic duplicate suffixes (e.g., "-0", "-1", etc.)
        candidate_clean = re.sub(r'-\d+$', '', candidate)
        
        # Direct string match first (most common case)
        if standard == candidate_clean:
            return True
            
        # Normalize both names for fallback matching
        std_norm = self.normalize_channel_name(standard.replace('-', ''))  
        cand_norm = self.normalize_channel_name(candidate_clean.replace('-', ''))
        
        # Check if they match after normalization
        return std_norm == cand_norm
        
    def parse_chbmit_summary(self, summary_path: str, filename: str) -> Dict:
        """
        Parse CHB-MIT summary file for seizure information
        
        Args:
            summary_path: Path to summary.txt file
            filename: Name of EDF file to look for
            
        Returns:
            Dictionary with seizure metadata
        """
        if not os.path.exists(summary_path):
            return {"seizures": 0, "times": []}
            
        try:
            with open(summary_path, 'r') as f:
                content = f.read()
                
            # Look for the specific file
            file_pattern = rf"File Name:\s*{re.escape(filename)}"
            match = re.search(file_pattern, content)
            
            if not match:
                return {"seizures": 0, "times": []}
                
            # Extract section for this file
            start_pos = match.start()
            next_file = re.search(r"File Name:", content[start_pos + 10:])
            end_pos = next_file.start() + start_pos + 10 if next_file else len(content)
            file_section = content[start_pos:end_pos]
            
            # Extract number of seizures
            seizure_match = re.search(r"Number of Seizures in File:\s*(\d+)", file_section)
            if not seizure_match:
                return {"seizures": 0, "times": []}
                
            num_seizures = int(seizure_match.group(1))
            seizure_times = []
            
            if num_seizures > 0:
                # Extract seizure times (convert from seconds to samples)
                start_times = re.findall(r"Seizure Start Time:\s*(\d+)", file_section)
                end_times = re.findall(r"Seizure End Time:\s*(\d+)", file_section)
                
                for start, end in zip(start_times, end_times):
                    start_sample = int(start) * self.sampling_rate
                    end_sample = int(end) * self.sampling_rate
                    seizure_times.append((start_sample, end_sample))
                    
            return {"seizures": num_seizures, "times": seizure_times}
            
        except Exception as e:
            print(f"Error parsing summary file {summary_path}: {e}")
            return {"seizures": 0, "times": []}
    
    def segment_signal(self, signal: np.ndarray, seizure_times: List[Tuple[int, int]], 
                      filename: str) -> List[Dict]:
        """
        Segment signal into fixed-length segments with labels
        
        Args:
            signal: Signal array (n_channels, n_samples)
            seizure_times: List of (start_sample, end_sample) tuples
            filename: Original filename for segment naming
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        n_samples = signal.shape[1]
        
        # Create non-overlapping segments
        for i in range(0, n_samples, self.segment_samples):
            if i + self.segment_samples > n_samples:
                break  # Skip incomplete segments
                
            segment_signal = signal[:, i:i + self.segment_samples]
            
            # Check if segment overlaps with any seizure
            label = 0
            for start_sample, end_sample in seizure_times:
                segment_start = i
                segment_end = i + self.segment_samples
                
                # Check for overlap
                if (segment_start < end_sample and segment_end > start_sample):
                    label = 1
                    break
                    
            segment = {
                "X": segment_signal.astype(np.float32),
                "y": label,
                "metadata": {
                    "filename": filename,
                    "segment_start": i,
                    "segment_end": i + self.segment_samples,
                    "sampling_rate": self.sampling_rate
                }
            }
            
            segments.append(segment)
            
        print(f"Created {len(segments)} segments from {filename}")
        if seizure_times:
            seizure_segments = sum(1 for s in segments if s["y"] == 1)
            print(f"  - {seizure_segments} seizure segments, {len(segments) - seizure_segments} non-seizure segments")
            
        return segments
        
    def process_edf_file(self, edf_path: str, output_dir: str, 
                        summary_path: Optional[str] = None) -> List[str]:
        """
        Process a single EDF file
        
        Args:
            edf_path: Path to EDF file
            output_dir: Output directory for pickle files
            summary_path: Path to summary file (for CHB-MIT data)
            
        Returns:
            List of created pickle file paths
        """
        print(f"Processing {edf_path}")
        
        # Load EDF file
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception as e:
            print(f"Error loading {edf_path}: {e}")
            return []
            
        print(f"Loaded EDF: {len(raw.ch_names)} channels, {raw.n_times/raw.info['sfreq']:.1f}s at {raw.info['sfreq']}Hz")
        
        # Extract/create bipolar channels
        try:
            signal, channel_names = self.extract_bipolar_channels(raw)
        except Exception as e:
            print(f"Error creating bipolar montage for {edf_path}: {e}")
            return []
            
        # Parse seizure information if summary provided
        filename = os.path.basename(edf_path)
        seizure_metadata = {"seizures": 0, "times": []}
        
        if summary_path:
            seizure_metadata = self.parse_chbmit_summary(summary_path, filename)
            
        # Segment the signal
        segments = self.segment_signal(signal, seizure_metadata["times"], filename)
        
        # Save segments as pickle files
        os.makedirs(output_dir, exist_ok=True)
        output_files = []
        
        filename_base = os.path.splitext(filename)[0]
        
        for i, segment in enumerate(segments):
            output_filename = os.path.join(output_dir, f"{filename_base}_segment_{i:04d}.pkl")
            
            with open(output_filename, 'wb') as f:
                pickle.dump({"X": segment["X"], "y": segment["y"]}, f)
                
            output_files.append(output_filename)
            
        print(f"Saved {len(output_files)} segments to {output_dir}")
        return output_files
        
    def process_chbmit_patient(self, patient_dir: str, output_dir: str) -> List[str]:
        """
        Process all EDF files for a CHB-MIT patient
        
        Args:
            patient_dir: Directory containing patient EDF files and summary
            output_dir: Output directory for pickle files
            
        Returns:
            List of created pickle file paths
        """
        patient_id = os.path.basename(patient_dir)
        print(f"Processing CHB-MIT patient: {patient_id}")
        
        # Find summary file
        summary_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")
        if not os.path.exists(summary_path):
            print(f"Warning: No summary file found at {summary_path}")
            summary_path = None
            
        # Process all EDF files
        edf_files = [f for f in os.listdir(patient_dir) if f.lower().endswith('.edf')]
        all_output_files = []
        
        for edf_file in tqdm(edf_files, desc=f"Processing {patient_id}"):
            edf_path = os.path.join(patient_dir, edf_file)
            output_files = self.process_edf_file(edf_path, output_dir, summary_path)
            all_output_files.extend(output_files)
            
        return all_output_files
        
    def process_palantir_files(self, input_dir: str, output_dir: str) -> List[str]:
        """
        Process Palantir EDF files (no seizure annotations assumed)
        
        Args:
            input_dir: Directory containing Palantir EDF files
            output_dir: Output directory for pickle files
            
        Returns:
            List of created pickle file paths  
        """
        print(f"Processing Palantir files from: {input_dir}")
        
        edf_files = [f for f in os.listdir(input_dir) if f.upper().endswith('.EDF')]
        all_output_files = []
        
        for edf_file in tqdm(edf_files, desc="Processing Palantir files"):
            edf_path = os.path.join(input_dir, edf_file)
            output_files = self.process_edf_file(edf_path, output_dir, summary_path=None)
            all_output_files.extend(output_files)
            
        return all_output_files


def main():
    parser = argparse.ArgumentParser(description="Unified EDF to Raw Pickle Preprocessing")
    parser.add_argument("--input_dir", required=True, help="Input directory containing EDF files")
    parser.add_argument("--output_dir", required=True, help="Output directory for pickle files")
    parser.add_argument("--data_type", choices=["chbmit", "palantir", "auto"], default="auto",
                       help="Type of data being processed")
    parser.add_argument("--sampling_rate", type=int, default=256, help="Sampling rate of input data")
    parser.add_argument("--segment_length", type=int, default=10, help="Segment length in seconds")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes")
    
    args = parser.parse_args()
    
    processor = UnifiedEDFProcessor(args.sampling_rate, args.segment_length)
    
    # Auto-detect data type if needed
    if args.data_type == "auto":
        if "chb" in args.input_dir.lower():
            args.data_type = "chbmit"
        elif "palantir" in args.input_dir.lower():
            args.data_type = "palantir" 
        else:
            print("Cannot auto-detect data type. Please specify --data_type")
            return
            
    print(f"Processing {args.data_type} data from {args.input_dir}")
    
    if args.data_type == "chbmit":
        # Process CHB-MIT data (assuming input_dir contains patient directories)
        patient_dirs = [d for d in os.listdir(args.input_dir) 
                       if os.path.isdir(os.path.join(args.input_dir, d)) and d.startswith('chb')]
        
        all_files = []
        for patient_dir in patient_dirs:
            patient_path = os.path.join(args.input_dir, patient_dir)
            output_files = processor.process_chbmit_patient(patient_path, args.output_dir)
            all_files.extend(output_files)
            
    elif args.data_type == "palantir":
        # Process Palantir data
        all_files = processor.process_palantir_files(args.input_dir, args.output_dir)
        
    print(f"Processing complete! Created {len(all_files)} segment files in {args.output_dir}")


if __name__ == "__main__":
    main()
