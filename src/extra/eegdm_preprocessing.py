"""
EEGDM-Compatible EEG Preprocessing Script (CORRECTED)

defaults to chb mit

Run like this:
python eegdm_preprocessing.py /path/to/file.EDF /tmp/test_output --verbose

This script implements the exact preprocessing pipeline used by EEGDM for both:
1. Temple University (TUEV) pretraining style
2. CHB-MIT fine-tuning style

CORRECTIONS from original:
1. Handles data that is ALREADY in bipolar montage (like CHB-MIT)
2. Properly segments data BEFORE resampling
3. Correct resampling that preserves temporal structure

All parameters are extracted directly from the EEGDM codebase to ensure compatibility.
"""

import mne
import numpy as np
import pickle
import os
import re
from scipy import signal
from typing import Optional, List, Dict, Tuple
import logging

# Suppress MNE verbose output
mne.set_log_level("CRITICAL")


class EEGDMPreprocessor:
    """
    EEG preprocessor that matches EEGDM's exact preprocessing pipeline.
    
    Based on analysis of EEGDM codebase:
    - src/preprocessing.py (TUEV preprocessing)
    - src/util.py (CHB-MIT preprocessing functions)
    - conf/preprocessing/pretrain.yaml (TUEV parameters)
    - conf/finetune/base_chbmit_bin_filt.yaml (CHB-MIT parameters)
    """
    
    # The 16 bipolar channels expected by EEGDM for CHB-MIT
    # These use the 10-10 naming convention (T7/T8/P7/P8)
    EXPECTED_BIPOLAR_CHANNELS = [
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2"
    ]
    
    # Mapping from various naming conventions to standard names
    # Handles both 10-10 (T7/T8/P7/P8) and 10-20 (T3/T4/T5/T6) conventions
    CHANNEL_NAME_ALIASES = {
        # 10-20 to 10-10 mapping for temporal channels
        "F7-T3": "F7-T7",
        "T3-T5": "T7-P7",
        "T5-O1": "P7-O1",
        "F8-T4": "F8-T8",
        "T4-T6": "T8-P8",
        "T6-O2": "P8-O2",
        # Also handle reverse (10-10 to itself)
        "F7-T7": "F7-T7",
        "T7-P7": "T7-P7",
        "P7-O1": "P7-O1",
        "F8-T8": "F8-T8",
        "T8-P8": "T8-P8",
        "P8-O2": "P8-O2",
        # Standard channels (no change needed)
        "FP1-F7": "FP1-F7",
        "FP2-F8": "FP2-F8",
        "FP1-F3": "FP1-F3",
        "F3-C3": "F3-C3",
        "C3-P3": "C3-P3",
        "P3-O1": "P3-O1",
        "FP2-F4": "FP2-F4",
        "F4-C4": "F4-C4",
        "C4-P4": "C4-P4",
        "P4-O2": "P4-O2",
    }
    
    def __init__(self, style: str = 'chbmit'):
        """
        Initialize the preprocessor.
        
        Args:
            style: 'chbmit' or 'tuev' preprocessing style
        """
        if style not in ['chbmit', 'tuev']:
            raise ValueError("style must be 'chbmit' or 'tuev'")
        
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # EXACT parameters from EEGDM codebase
        if style == 'tuev':
            # From conf/preprocessing/pretrain.yaml
            self.bandpass_l = 0.1
            self.bandpass_h = 75
            self.notch_freq = 50  # European standard
            self.target_sfreq = 200
            self.segment_seconds = 5.0
        else:  # chbmit
            # From src/util.py data_transform_chbmit_filt function
            self.bandpass_l = 0.5
            self.bandpass_h = 75
            self.notch_freq = 60  # US standard
            self.target_sfreq = 200
            self.segment_seconds = 10.0
    
    def mu_law(self, x: np.ndarray, mu: int = 255) -> np.ndarray:
        """
        Apply µ-law companding.
        
        Exact implementation from src/util.py
        """
        return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    
    def staged_mu_law(self, x: np.ndarray, mu: int = 255, scale: float = 1) -> np.ndarray:
        """
        Apply staged µ-law companding (only to values outside [-1, 1]).
        
        Exact implementation from src/util.py
        """
        x = scale * x.copy()  # Make a copy to avoid modifying input
        _x = self.mu_law(x, mu=mu)
        mask_pos = x > 1
        mask_neg = x < -1
        x[mask_pos] = _x[mask_pos]
        x[mask_neg] = _x[mask_neg]
        return x / scale
    
    def div_100_staged_mu_law(self, x: np.ndarray, mu: int = 255) -> np.ndarray:
        """
        Divide by 100 then apply staged µ-law.
        
        Exact implementation from src/util.py for CHB-MIT
        """
        return self.staged_mu_law(x / 100, mu=mu)
    
    def parse_chbmit_summary(self, summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        Parse CHB-MIT summary.txt file to extract seizure timing information.
        
        Args:
            summary_path: Path to summary.txt file
            
        Returns:
            Dictionary mapping EDF filenames to list of (start_sec, end_sec) tuples
        """
        seizure_info = {}
        
        try:
            with open(summary_path, 'r') as f:
                content = f.read()
            
            # Split into sections for each file
            file_sections = re.split(r'File Name: ', content)[1:]  # Skip header
            
            for section in file_sections:
                lines = section.strip().split('\n')
                if not lines:
                    continue
                
                # Extract filename from first line
                filename = lines[0].strip()
                
                # Initialize seizure list for this file
                seizure_info[filename] = []
                
                # Look for seizure information
                for i, line in enumerate(lines):
                    if line.startswith('Number of Seizures in File:'):
                        num_seizures = int(line.split(':')[1].strip())
                        
                        if num_seizures > 0:
                            # Look for seizure timing on following lines
                            j = i + 1
                            while j < len(lines) and j < i + 1 + (num_seizures * 2):
                                if j < len(lines) and lines[j].startswith('Seizure Start Time:'):
                                    start_time = int(lines[j].split(':')[1].strip().split()[0])
                                if j + 1 < len(lines) and lines[j + 1].startswith('Seizure End Time:'):
                                    end_time = int(lines[j + 1].split(':')[1].strip().split()[0])
                                    seizure_info[filename].append((start_time, end_time))
                                    j += 2
                                else:
                                    j += 1
                        break
            
            self.logger.info(f"Parsed seizure info for {len(seizure_info)} files from {summary_path}")
            seizure_files = [f for f, seizures in seizure_info.items() if seizures]
            self.logger.info(f"Found {len(seizure_files)} files with seizures")
            
        except Exception as e:
            self.logger.error(f"Error parsing summary file {summary_path}: {e}")
            return {}
        
        return seizure_info
    
    def assign_segment_labels(self, filename: str, seizure_intervals: List[Tuple[int, int]], 
                             num_segments: int) -> Dict[int, int]:
        """
        Assign labels to segments based on seizure timing.
        
        Args:
            filename: EDF filename for logging
            seizure_intervals: List of (start_sec, end_sec) tuples
            num_segments: Total number of segments
            
        Returns:
            Dictionary mapping segment index to label (0=non-seizure, 1=seizure)
        """
        segment_labels = {}
        
        for segment_idx in range(num_segments):
            segment_start_sec = segment_idx * self.segment_seconds
            segment_end_sec = (segment_idx + 1) * self.segment_seconds
            
            # Check if this segment overlaps with any seizure
            is_seizure = False
            for seizure_start, seizure_end in seizure_intervals:
                # Check for overlap: segment overlaps seizure if:
                # segment_start < seizure_end AND segment_end > seizure_start
                if segment_start_sec < seizure_end and segment_end_sec > seizure_start:
                    is_seizure = True
                    break
            
            segment_labels[segment_idx] = 1 if is_seizure else 0
        
        # Log seizure segment statistics
        seizure_segments = sum(1 for label in segment_labels.values() if label == 1)
        if seizure_segments > 0:
            self.logger.info(f"{filename}: {seizure_segments}/{num_segments} segments labeled as seizure")
        
        return segment_labels
    
    def find_summary_file(self, edf_path: str) -> Optional[str]:
        """
        Find the corresponding summary.txt file for an EDF file.
        
        Args:
            edf_path: Path to EDF file
            
        Returns:
            Path to summary.txt file, or None if not found
        """
        edf_dir = os.path.dirname(edf_path)
        edf_basename = os.path.basename(edf_path)
        
        # Extract patient ID (e.g., 'chb01' from 'chb01_03.edf')
        patient_match = re.match(r'(chb\d+)_', edf_basename)
        if not patient_match:
            return None
        
        patient_id = patient_match.group(1)
        summary_filename = f"{patient_id}-summary.txt"
        summary_path = os.path.join(edf_dir, summary_filename)
        
        if os.path.exists(summary_path):
            return summary_path
        
        return None
    
    def is_already_bipolar(self, raw: mne.io.Raw) -> bool:
        """
        Check if data is already in bipolar montage format.
        
        Bipolar channels contain '-' in their names (e.g., 'FP1-F7')
        """
        bipolar_count = sum(1 for ch in raw.ch_names if '-' in ch)
        total_count = len(raw.ch_names)
        
        # Consider it bipolar if majority of channels have '-'
        is_bipolar = bipolar_count > total_count / 2
        
        self.logger.info(f"Channel format detection: {bipolar_count}/{total_count} channels "
                        f"are bipolar format → {'BIPOLAR' if is_bipolar else 'ELECTRODE'}")
        return is_bipolar
    
    def standardize_channel_name(self, ch_name: str) -> str:
        """
        Standardize a channel name to the expected format.
        
        Handles case variations and electrode naming conventions.
        """
        import re

        # Uppercase and strip whitespace
        ch_upper = ch_name.upper().strip()

        # Remove MNE's duplicate channel suffixes (-0, -1, -2, etc.)
        # Example: 'T8-P8-0' becomes 'T8-P8'
        ch_upper = re.sub(r'-(\d+)$', '', ch_upper)
        
        # Replace common variations
        ch_upper = ch_upper.replace("EEG ", "").replace("EEG-", "")
        
        # Check if it's in our alias mapping
        if ch_upper in self.CHANNEL_NAME_ALIASES:
            return self.CHANNEL_NAME_ALIASES[ch_upper]
        
        return ch_upper
    
    def select_bipolar_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Select and reorder bipolar channels from already-bipolar data.
        
        Args:
            raw: MNE Raw object with bipolar montage channels
            
        Returns:
            MNE Raw object with 16 selected channels in correct order
        """
        # Create mapping from standardized names to original names
        original_to_standard = {}
        standard_to_original = {}
        
        for ch in raw.ch_names:
            std_name = self.standardize_channel_name(ch)
            original_to_standard[ch] = std_name
            standard_to_original[std_name] = ch
        
        self.logger.info(f"Available channels (standardized): {list(standard_to_original.keys())}")
        
        # Find which expected channels are available
        channels_to_pick = []
        missing_channels = []
        
        for expected_ch in self.EXPECTED_BIPOLAR_CHANNELS:
            if expected_ch in standard_to_original:
                channels_to_pick.append(standard_to_original[expected_ch])
            else:
                missing_channels.append(expected_ch)
        
        if missing_channels:
            self.logger.warning(f"Missing channels: {missing_channels}")
        
        if len(channels_to_pick) == 0:
            raise ValueError("No expected bipolar channels found in data!")
        
        self.logger.info(f"Selecting {len(channels_to_pick)}/16 channels")
        
        # Pick the available channels
        raw_selected = raw.copy().pick_channels(channels_to_pick, ordered=False)
        
        # Rename to standardized names
        rename_dict = {ch: original_to_standard[ch] for ch in raw_selected.ch_names}
        raw_selected.rename_channels(rename_dict)
        
        # Reorder to match expected order
        available_in_order = [ch for ch in self.EXPECTED_BIPOLAR_CHANNELS 
                             if ch in raw_selected.ch_names]
        raw_selected.reorder_channels(available_in_order)
        
        self.logger.info(f"Final channel order: {raw_selected.ch_names}")
        return raw_selected
    
    def create_bipolar_montage(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Create bipolar montage from individual electrode channels.
        
        Args:
            raw: MNE Raw object with individual electrode channels
            
        Returns:
            MNE Raw object with bipolar montage channels
        """
        # Define the bipolar pairs we want to create
        bipolar_pairs = [
            ('FP1', 'F7'),   # FP1-F7
            ('F7', 'T7'),    # F7-T7
            ('T7', 'P7'),    # T7-P7
            ('P7', 'O1'),    # P7-O1
            ('FP2', 'F8'),   # FP2-F8
            ('F8', 'T8'),    # F8-T8
            ('T8', 'P8'),    # T8-P8
            ('P8', 'O2'),    # P8-O2
            ('FP1', 'F3'),   # FP1-F3
            ('F3', 'C3'),    # F3-C3
            ('C3', 'P3'),    # C3-P3
            ('P3', 'O1'),    # P3-O1
            ('FP2', 'F4'),   # FP2-F4
            ('F4', 'C4'),    # F4-C4
            ('C4', 'P4'),    # C4-P4
            ('P4', 'O2')     # P4-O2
        ]
        
        # Build mapping from uppercase names to original channel names
        ch_name_map = {}
        for ch in raw.ch_names:
            ch_upper = ch.upper().strip()
            # Handle common variations (Fp vs FP, etc.)
            ch_normalized = ch_upper.replace("FP", "FP").replace("EEG ", "").replace("EEG-", "")
            ch_name_map[ch_normalized] = ch
        
        self.logger.info(f"Available electrodes: {list(ch_name_map.keys())}")
        
        # Create bipolar channels
        bipolar_data = []
        bipolar_names = []
        
        for anode, cathode in bipolar_pairs:
            anode_key = anode.upper()
            cathode_key = cathode.upper()
            
            if anode_key in ch_name_map and cathode_key in ch_name_map:
                # Get original channel names
                anode_orig = ch_name_map[anode_key]
                cathode_orig = ch_name_map[cathode_key]
                
                # Get data for both electrodes
                anode_data = raw.get_data(picks=[anode_orig])
                cathode_data = raw.get_data(picks=[cathode_orig])
                
                # Create bipolar channel (anode - cathode)
                bipolar_signal = anode_data[0] - cathode_data[0]
                bipolar_data.append(bipolar_signal)
                bipolar_names.append(f"{anode}-{cathode}")
            else:
                missing = []
                if anode_key not in ch_name_map:
                    missing.append(anode)
                if cathode_key not in ch_name_map:
                    missing.append(cathode)
                self.logger.warning(f"Cannot create {anode}-{cathode}: missing {missing}")
        
        if not bipolar_data:
            raise ValueError("No bipolar channels could be created from available electrodes")
        
        # Create new Raw object with bipolar montage
        bipolar_array = np.array(bipolar_data)
        info = mne.create_info(
            ch_names=bipolar_names,
            sfreq=raw.info['sfreq'],
            ch_types=['eeg'] * len(bipolar_names)
        )
        
        raw_bipolar = mne.io.RawArray(bipolar_array, info)
        
        self.logger.info(f"Created {len(bipolar_names)} bipolar channels: {bipolar_names}")
        return raw_bipolar
    
    def segment_data(self, data: np.ndarray, original_sfreq: float) -> List[np.ndarray]:
        """
        Segment data into fixed-length segments and resample each.
        
        Args:
            data: EEG data array (n_channels, n_samples)
            original_sfreq: Original sampling frequency
            
        Returns:
            List of resampled segments
        """
        # Calculate segment sizes
        segment_samples_in = int(self.segment_seconds * original_sfreq)
        segment_samples_out = int(self.segment_seconds * self.target_sfreq)
        
        n_samples = data.shape[1]
        n_segments = n_samples // segment_samples_in
        
        if n_segments == 0:
            self.logger.warning(f"Data too short for {self.segment_seconds}s segments. "
                              f"Got {n_samples/original_sfreq:.2f}s")
            # Process whatever we have
            n_segments = 1
            segment_samples_in = n_samples
        
        segments = []
        for i in range(n_segments):
            start = i * segment_samples_in
            end = start + segment_samples_in
            
            if end > n_samples:
                break
                
            segment = data[:, start:end]
            
            # Resample this segment to target samples
            segment_resampled = signal.resample(segment, segment_samples_out, axis=1)
            segments.append(segment_resampled)
        
        self.logger.info(f"Created {len(segments)} segments of {self.segment_seconds}s "
                        f"({segment_samples_out} samples each at {self.target_sfreq}Hz)")
        return segments
    
    def apply_filters(self, data: np.ndarray, sfreq: float) -> np.ndarray:
        """
        Apply bandpass and notch filters to data.
        
        Args:
            data: EEG data array (n_channels, n_samples)
            sfreq: Sampling frequency of the data
            
        Returns:
            Filtered data array
        """
        # Bandpass filter
        data_filtered = mne.filter.filter_data(
            data,
            sfreq,
            l_freq=self.bandpass_l,
            h_freq=self.bandpass_h,
            verbose=False
        )
        
        # Notch filter
        data_notched = mne.filter.notch_filter(
            data_filtered,
            sfreq,
            self.notch_freq,
            verbose=False
        )
        
        return data_notched
    
    def preprocess_chbmit_style(self, raw: mne.io.Raw) -> List[np.ndarray]:
        """
        Apply CHB-MIT-style preprocessing.
        
        Pipeline:
        1. Segment into 10-second chunks
        2. Resample each segment to 200 Hz (2000 samples)
        3. Apply bandpass filter (0.5-75 Hz)
        4. Apply notch filter (60 Hz)
        5. Apply div_100_staged_mu_law normalization
        
        Returns:
            List of preprocessed segments
        """
        original_sfreq = raw.info['sfreq']
        data = raw.get_data()
        
        self.logger.info(f"CHB-MIT preprocessing: {data.shape[1]/original_sfreq:.1f}s "
                        f"at {original_sfreq}Hz")
        
        # Step 1-2: Segment and resample
        segments = self.segment_data(data, original_sfreq)
        
        # Process each segment
        processed_segments = []
        for i, segment in enumerate(segments):
            # Step 3-4: Apply filters (now at target_sfreq)
            segment_filtered = self.apply_filters(segment, self.target_sfreq)
            
            # Step 5: Apply normalization
            segment_normalized = self.div_100_staged_mu_law(segment_filtered, mu=255)
            
            processed_segments.append(segment_normalized.astype(np.float32))
        
        return processed_segments
    
    def preprocess_tuev_style(self, raw: mne.io.Raw) -> List[np.ndarray]:
        """
        Apply TUEV-style preprocessing.
        
        Pipeline:
        1. Apply bandpass filter (0.1-75 Hz)
        2. Apply notch filter (50 Hz)
        3. Resample to 200 Hz
        4. Scale by 1e4
        5. Apply staged_mu_law normalization
        6. Segment into 5-second chunks
        
        Returns:
            List of preprocessed segments
        """
        original_sfreq = raw.info['sfreq']
        
        self.logger.info(f"TUEV preprocessing: {raw.n_times/original_sfreq:.1f}s "
                        f"at {original_sfreq}Hz")
        
        # Step 1-2: Apply filters at original sampling rate
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=self.bandpass_l, h_freq=self.bandpass_h, verbose=False)
        raw_filtered.notch_filter(self.notch_freq, verbose=False)
        
        # Step 3: Resample
        raw_filtered.resample(self.target_sfreq, verbose=False)
        
        # Get data
        data = raw_filtered.get_data()
        
        # Step 4: Scale
        data = data * 1e4
        
        # Step 5: Apply normalization
        data = self.staged_mu_law(data, mu=255)
        
        # Step 6: Segment
        segment_samples = int(self.segment_seconds * self.target_sfreq)
        n_segments = data.shape[1] // segment_samples
        
        segments = []
        for i in range(n_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = data[:, start:end].astype(np.float32)
            segments.append(segment)
        
        self.logger.info(f"Created {len(segments)} segments of {self.segment_seconds}s")
        return segments
    
    def process_edf_file(self,
                        edf_path: str,
                        output_dir: str,
                        labels: Optional[Dict[int, int]] = None) -> List[str]:
        """
        Process a single EDF file and save in EEGDM-compatible format.
        
        Args:
            edf_path: Path to input EDF file
            output_dir: Directory to save processed files
            labels: Optional dictionary mapping segment indices to labels
                    If None, will try to auto-detect from summary.txt files
            
        Returns:
            List of output file paths
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load EDF file
        self.logger.info(f"Loading {edf_path}")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        self.logger.info(f"Original data: {len(raw.ch_names)} channels, "
                        f"{raw.n_times/raw.info['sfreq']:.1f}s at {raw.info['sfreq']}Hz")
        
        # Check if data is already bipolar or needs conversion
        if self.is_already_bipolar(raw):
            # Data is already bipolar - just select the right channels
            raw_bipolar = self.select_bipolar_channels(raw)
        else:
            # Data has individual electrodes - create bipolar montage
            raw_bipolar = self.create_bipolar_montage(raw)
        
        # Apply preprocessing based on style
        if self.style == 'chbmit':
            processed_segments = self.preprocess_chbmit_style(raw_bipolar)
        else:  # tuev
            processed_segments = self.preprocess_tuev_style(raw_bipolar)
        
        # If no labels provided, try to find and parse summary file
        if labels is None:
            summary_path = self.find_summary_file(edf_path)
            if summary_path:
                self.logger.info(f"Found summary file: {summary_path}")
                seizure_info = self.parse_chbmit_summary(summary_path)
                filename = os.path.basename(edf_path)
                
                if filename in seizure_info:
                    seizure_intervals = seizure_info[filename]
                    labels = self.assign_segment_labels(filename, seizure_intervals, len(processed_segments))
                else:
                    self.logger.warning(f"No seizure info found for {filename} in summary file")
                    labels = {}
            else:
                self.logger.info("No summary file found, using default labels (all non-seizure)")
                labels = {}
        
        # Save segments
        output_files = []
        filename_base = os.path.splitext(os.path.basename(edf_path))[0]
        
        for i, segment in enumerate(processed_segments):
            output_filename = os.path.join(output_dir, f"{filename_base}_segment_{i:04d}.pkl")
            
            # Create data dictionary
            if self.style == 'chbmit':
                data_dict = {
                    "X": segment,
                    "y": labels.get(i, 0) if labels else 0
                }
            else:  # tuev
                data_dict = {
                    "data": segment,
                    "label": labels.get(i, 0) if labels else 0
                }
            
            with open(output_filename, "wb") as f:
                pickle.dump(data_dict, f)
            
            output_files.append(output_filename)
        
        self.logger.info(f"Saved {len(output_files)} segments to {output_dir}")
        return output_files
    
    def process_directory(self,
                         input_dir: str,
                         output_dir: str,
                         labels_dict: Optional[Dict[str, Dict[int, int]]] = None) -> List[str]:
        """
        Process all EDF files in a directory.
        
        Args:
            input_dir: Directory containing EDF files
            output_dir: Directory to save processed files
            labels_dict: Optional dict mapping filenames to segment label dicts
            
        Returns:
            List of all output file paths
        """
        all_output_files = []
        
        edf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.edf')]
        self.logger.info(f"Found {len(edf_files)} EDF files in {input_dir}")
        
        for filename in sorted(edf_files):
            edf_path = os.path.join(input_dir, filename)
            file_labels = labels_dict.get(filename) if labels_dict else None
            
            try:
                output_files = self.process_edf_file(
                    edf_path=edf_path,
                    output_dir=output_dir,
                    labels=file_labels
                )
                all_output_files.extend(output_files)
            except Exception as e:
                self.logger.error(f"Error processing {filename}: {e}")
                continue
        
        self.logger.info(f"Processed {len(all_output_files)} total segments")
        return all_output_files


def main():
    """Example usage of the EEGDMPreprocessor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EEGDM-compatible EEG preprocessing")
    parser.add_argument("input_path", help="Input EDF file or directory")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--style", choices=['chbmit', 'tuev'], default='chbmit',
                       help="Preprocessing style (default: chbmit)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize preprocessor
    processor = EEGDMPreprocessor(style=args.style)
    
    # Process input
    if os.path.isfile(args.input_path):
        output_files = processor.process_edf_file(args.input_path, args.output_dir)
        print(f"Processed 1 file → {len(output_files)} segments")
    elif os.path.isdir(args.input_path):
        output_files = processor.process_directory(args.input_path, args.output_dir)
        print(f"Processed directory → {len(output_files)} total segments")
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")
        return 1
    
    # Print summary
    if output_files:
        print(f"\nOutput files saved to: {args.output_dir}")
        print(f"First file: {output_files[0]}")
        if len(output_files) > 1:
            print(f"Last file:  {output_files[-1]}")
    
    return 0


if __name__ == "__main__":
    exit(main())

# import mne
# import numpy as np
# import pickle
# import os
# from scipy import signal
# from typing import Optional, List, Dict, Tuple
# import logging

# # Suppress MNE verbose output
# mne.set_log_level("CRITICAL")

# class EEGDMPreprocessor:
#     """
#     EEG preprocessor that matches EEGDM's exact preprocessing pipeline.
    
#     Based on analysis of EEGDM codebase:
#     - src/preprocessing.py (TUEV preprocessing)
#     - src/util.py (CHB-MIT preprocessing functions)
#     - conf/preprocessing/pretrain.yaml (TUEV parameters)
#     - conf/finetune/base_chbmit_bin_filt.yaml (CHB-MIT parameters)
#     """
    
#     # EXACT channel mapping from EEGDM CHB-MIT configuration
#     # use_cond: [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21]
#     EEGDM_CHBMIT_CHANNELS = [
#         "FP1-F7",  # index 0
#         "F7-T3",   # index 1  
#         "T3-T5",   # index 2
#         "T5-O1",   # index 3
#         "FP2-F8",  # index 4
#         "F8-T4",   # index 5
#         "T4-T6",   # index 6
#         "T6-O2",   # index 7
#         "FP1-F3",  # index 14
#         "F3-C3",   # index 15
#         "C3-P3",   # index 16
#         "P3-O1",   # index 17
#         "FP2-F4",  # index 18
#         "F4-C4",   # index 19
#         "C4-P4",   # index 20
#         "P4-O2"    # index 21
#     ]
    
#     # User's channel mapping to EEGDM pattern
#     USER_TO_EEGDM_MAPPING = {
#         "FP1-F7": "FP1-F7",    # Channel 1 → index 0
#         "F7-T7": "F7-T3",      # Channel 2 → index 1 (T7→T3)
#         "T7-P7": "T3-T5",      # Channel 3 → index 2 (T7→T3, P7→T5)
#         "P7-O1": "T5-O1",      # Channel 4 → index 3 (P7→T5)
#         "FP1-F3": "FP1-F3",    # Channel 6 → index 14
#         "F3-C3": "F3-C3",      # Channel 7 → index 15
#         "C3-P3": "C3-P3",      # Channel 8 → index 16
#         "P3-O1": "P3-O1",      # Channel 9 → index 17
#         "FP2-F4": "FP2-F4",    # Channel 14 → index 18
#         "F4-C4": "F4-C4",      # Channel 15 → index 19
#         "C4-P4": "C4-P4",      # Channel 16 → index 20
#         "P4-O2": "P4-O2",      # Channel 17 → index 21
#         "FP2-F8": "FP2-F8",    # Channel 19 → index 4
#         "F8-T8": "F8-T4",      # Channel 20 → index 5 (T8→T4)
#         "T8-P8": "T4-T6",      # Channel 21 → index 6 (T8→T4, P8→T6)
#         "P8-O2": "T6-O2"       # Channel 22 → index 7 (P8→T6)
#     }
    
#     def __init__(self, style: str = 'chbmit'):
#         """
#         Initialize the preprocessor.
        
#         Args:
#             style: 'chbmit' or 'tuev' preprocessing style
#         """
#         if style not in ['chbmit', 'tuev']:
#             raise ValueError("style must be 'chbmit' or 'tuev'")
        
#         self.style = style
#         self.logger = logging.getLogger(__name__)
        
#         # EXACT parameters from EEGDM codebase
#         if style == 'tuev':
#             # From conf/preprocessing/pretrain.yaml
#             self.bandpass_l = 0.1
#             self.bandpass_h = 75
#             self.notch_freq = 50  # European standard
#             self.sfreq = 200
#             self.scale = 1e4  # Scale up by 1e4 (equivalent to scaling down µV by 100)
#         else:  # chbmit
#             # From src/util.py data_transform_chbmit_filt function
#             self.bandpass_l = 0.5
#             self.bandpass_h = 75
#             self.notch_freq = 60  # US standard
#             self.sfreq = 200
#             self.scale_factor = 100  # Divide by 100
    
#     def mu_law(self, x: np.ndarray, mu: int = 255) -> np.ndarray:
#         """
#         Apply µ-law companding. 
        
#         Exact implementation from src/util.py
#         """
#         return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    
#     def staged_mu_law(self, x: np.ndarray, mu: int = 255, scale: float = 1) -> np.ndarray:
#         """
#         Apply staged µ-law companding (only to values outside [-1, 1]).
        
#         Exact implementation from src/util.py
#         """
#         x = scale * x
#         _x = self.mu_law(x, mu=mu)
#         x[x > 1] = _x[x > 1]
#         x[x < -1] = _x[x < -1]
#         return x / scale
    
#     def div_100_staged_mu_law(self, x: np.ndarray, mu: int = 255) -> np.ndarray:
#         """
#         Divide by 100 then apply staged µ-law.
        
#         Exact implementation from src/util.py
#         """
#         return self.staged_mu_law(x / 100, mu=mu)
    
#     def create_bipolar_montage(self, raw: mne.io.Raw) -> mne.io.Raw:
#         """
#         Create bipolar montage from individual electrode channels.
        
#         Args:
#             raw: MNE Raw object with individual electrode channels
            
#         Returns:
#             MNE Raw object with bipolar montage channels
#         """
#         # Define the bipolar montage we want to create
#         # This matches the 16 channels expected by EEGDM
#         bipolar_montage = [
#             ('FP1', 'F7'),   # FP1-F7
#             ('F7', 'T7'),    # F7-T7  
#             ('T7', 'P7'),    # T7-P7
#             ('P7', 'O1'),    # P7-O1
#             ('FP2', 'F8'),   # FP2-F8
#             ('F8', 'T8'),    # F8-T8
#             ('T8', 'P8'),    # T8-P8
#             ('P8', 'O2'),    # P8-O2
#             ('FP1', 'F3'),   # FP1-F3
#             ('F3', 'C3'),    # F3-C3
#             ('C3', 'P3'),    # C3-P3
#             ('P3', 'O1'),    # P3-O1
#             ('FP2', 'F4'),   # FP2-F4
#             ('F4', 'C4'),    # F4-C4
#             ('C4', 'P4'),    # C4-P4
#             ('P4', 'O2')     # P4-O2
#         ]
        
#         # Handle channel name variations (some systems use Fp instead of FP)
#         available_channels = [ch.upper() for ch in raw.ch_names]
#         channel_mapping = {}
#         for ch in raw.ch_names:
#             ch_upper = ch.upper()
#             if ch_upper.startswith('FP'):
#                 # Map both Fp1 and FP1 to FP1, etc.
#                 standard_name = ch_upper.replace('FP', 'FP')
#                 channel_mapping[ch] = standard_name
#             else:
#                 channel_mapping[ch] = ch_upper
        
#         # Create a copy and rename channels to standardized format
#         raw_copy = raw.copy()
#         raw_copy.rename_channels(channel_mapping)
        
#         # Check which electrodes are available
#         available_electrodes = set(raw_copy.ch_names)
#         self.logger.info(f"Available electrodes: {sorted(available_electrodes)}")
        
#         # Create bipolar channels
#         bipolar_data = []
#         bipolar_names = []
        
#         for anode, cathode in bipolar_montage:
#             if anode in available_electrodes and cathode in available_electrodes:
#                 # Get data for both electrodes
#                 anode_data, _ = raw_copy[anode, :]
#                 cathode_data, _ = raw_copy[cathode, :]
                
#                 # Create bipolar channel (anode - cathode)
#                 bipolar_signal = anode_data - cathode_data
#                 bipolar_data.append(bipolar_signal[0])  # Remove extra dimension
#                 bipolar_names.append(f"{anode}-{cathode}")
#             else:
#                 missing = []
#                 if anode not in available_electrodes:
#                     missing.append(anode)
#                 if cathode not in available_electrodes:
#                     missing.append(cathode)
#                 self.logger.warning(f"Cannot create {anode}-{cathode}: missing {missing}")
        
#         if not bipolar_data:
#             raise ValueError("No bipolar channels could be created from available electrodes")
        
#         # Create new Raw object with bipolar montage
#         bipolar_array = np.array(bipolar_data)
#         info = mne.create_info(
#             ch_names=bipolar_names,
#             sfreq=raw.info['sfreq'],
#             ch_types=['eeg'] * len(bipolar_names)
#         )
        
#         raw_bipolar = mne.io.RawArray(bipolar_array, info)
        
#         self.logger.info(f"Created {len(bipolar_names)} bipolar channels: {bipolar_names}")
#         return raw_bipolar
    
#     def map_channels_to_eegdm(self, raw: mne.io.Raw) -> mne.io.Raw:
#         """
#         Map bipolar montage channels to EEGDM's expected CHB-MIT pattern.
        
#         Args:
#             raw: MNE Raw object with bipolar montage channels
            
#         Returns:
#             MNE Raw object with EEGDM-compatible channels
#         """
#         # Channel name mapping for EEGDM compatibility
#         # T7/T8 → T3/T4 and P7/P8 → T5/T6
#         channel_name_mapping = {
#             'FP1-F7': 'FP1-F7',
#             'F7-T7': 'F7-T3',     # T7 → T3
#             'T7-P7': 'T3-T5',     # T7 → T3, P7 → T5  
#             'P7-O1': 'T5-O1',     # P7 → T5
#             'FP2-F8': 'FP2-F8',
#             'F8-T8': 'F8-T4',     # T8 → T4
#             'T8-P8': 'T4-T6',     # T8 → T4, P8 → T6
#             'P8-O2': 'T6-O2',     # P8 → T6
#             'FP1-F3': 'FP1-F3',
#             'F3-C3': 'F3-C3',
#             'C3-P3': 'C3-P3',
#             'P3-O1': 'P3-O1',
#             'FP2-F4': 'FP2-F4',
#             'F4-C4': 'F4-C4',
#             'C4-P4': 'C4-P4',
#             'P4-O2': 'P4-O2'
#         }
        
#         # Get available channels that we can map
#         available_channels = raw.ch_names
#         mappable_channels = []
        
#         for user_ch, eegdm_ch in channel_name_mapping.items():
#             if user_ch in available_channels:
#                 mappable_channels.append(user_ch)
#             else:
#                 self.logger.warning(f"Channel {user_ch} not found in data")
        
#         if len(mappable_channels) < 16:
#             self.logger.warning(f"Only found {len(mappable_channels)}/16 expected channels")
        
#         # Select mappable channels
#         raw_selected = raw.pick_channels(mappable_channels, ordered=False)
        
#         # Rename channels to EEGDM format
#         mapping_dict = {}
#         for ch in raw_selected.ch_names:
#             if ch in channel_name_mapping:
#                 mapping_dict[ch] = channel_name_mapping[ch]
        
#         raw_selected.rename_channels(mapping_dict)
        
#         # Reorder to match EEGDM's expected order
#         eegdm_channels_present = [ch for ch in self.EEGDM_CHBMIT_CHANNELS 
#                                   if ch in raw_selected.ch_names]
#         raw_selected.reorder_channels(eegdm_channels_present)
        
#         self.logger.info(f"Mapped {len(eegdm_channels_present)} channels to EEGDM format")
#         return raw_selected
    
#     def tuev_style_preprocessing(self, raw: mne.io.Raw) -> np.ndarray:
#         """
#         Apply TUEV-style preprocessing.
        
#         Based on src/preprocessing.py and conf/preprocessing/pretrain.yaml
#         """
#         # Apply filters
#         raw_filtered = raw.copy()
#         raw_filtered.filter(l_freq=self.bandpass_l, h_freq=self.bandpass_h, verbose=False)
#         raw_filtered.notch_filter(self.notch_freq, verbose=False)
        
#         # Resample
#         raw_filtered.resample(self.sfreq, n_jobs=1, verbose=False)
        
#         # Get data and apply scaling
#         data, times = raw_filtered[:]
#         data *= self.scale
        
#         # Apply µ-law companding
#         data = self.staged_mu_law(data, mu=255)
        
#         return data
    
#     def chbmit_style_preprocessing(self, raw: mne.io.Raw) -> np.ndarray:
#         """
#         Apply CHB-MIT-style preprocessing.
        
#         Based on src/util.py data_transform_chbmit_filt function
#         """
#         # Get data
#         data, _ = raw[:]
        
#         # Step 1: Resample to 2000 samples (10 seconds at 200 Hz)
#         data_resampled = signal.resample(data, 2000, axis=1)
        
#         # Step 2: Apply bandpass filter (0.5-75 Hz)
#         data_filtered = mne.filter.filter_data(
#             data_resampled, 
#             self.sfreq,  # 200 Hz
#             l_freq=self.bandpass_l,  # 0.5 Hz
#             h_freq=self.bandpass_h,  # 75 Hz
#             verbose=False
#         )
        
#         # Step 3: Apply notch filter (60 Hz)
#         data_notched = mne.filter.notch_filter(
#             data_filtered,
#             self.sfreq,  # 200 Hz
#             self.notch_freq,  # 60 Hz
#             verbose=False
#         )
        
#         # Step 4: Apply div_100_staged_mu_law
#         data_final = self.div_100_staged_mu_law(data_notched, mu=255)
        
#         return data_final
    
#     def process_edf_file(self, 
#                         edf_path: str, 
#                         output_dir: str, 
#                         labels: Optional[Dict] = None,
#                         segment_length: float = 10.0) -> List[str]:
#         """
#         Process a single EDF file and save in EEGDM-compatible format.
        
#         Args:
#             edf_path: Path to input EDF file
#             output_dir: Directory to save processed files
#             labels: Optional dictionary mapping segment indices to labels
#             segment_length: Length of segments in seconds
            
#         Returns:
#             List of output file paths
#         """
#         # Ensure output directory exists
#         os.makedirs(output_dir, exist_ok=True)
        
#         # Load EDF file
#         self.logger.info(f"Loading {edf_path}")
#         raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
#         # Create bipolar montage from individual electrodes
#         raw_bipolar = self.create_bipolar_montage(raw)
        
#         # Map channels to EEGDM format
#         raw_mapped = self.map_channels_to_eegdm(raw_bipolar)
        
#         # Apply preprocessing
#         if self.style == 'tuev':
#             processed_data = self.tuev_style_preprocessing(raw_mapped)
#         else:
#             processed_data = self.chbmit_style_preprocessing(raw_mapped)
        
#         # Create segments
#         output_files = []
#         filename_base = os.path.splitext(os.path.basename(edf_path))[0]
        
#         if self.style == 'chbmit':
#             # CHB-MIT style: single 10-second segment
#             segment_idx = 0
#             output_filename = os.path.join(output_dir, f"{filename_base}_segment_{segment_idx}.pkl")
            
#             # Create data dictionary in CHB-MIT format
#             data_dict = {
#                 "X": processed_data.astype(np.float32),
#                 "y": labels.get(segment_idx, 0) if labels else 0  # Default to 0 (non-seizure)
#             }
            
#             with open(output_filename, "wb") as f:
#                 pickle.dump(data_dict, f)
            
#             output_files.append(output_filename)
#             self.logger.info(f"Saved CHB-MIT format: {output_filename}")
            
#         else:
#             # TUEV style: multiple 5-second segments  
#             segment_samples = int(segment_length * self.sfreq)
#             n_segments = processed_data.shape[1] // segment_samples
            
#             for i in range(n_segments):
#                 start_idx = i * segment_samples
#                 end_idx = start_idx + segment_samples
#                 segment_data = processed_data[:, start_idx:end_idx]
                
#                 output_filename = os.path.join(output_dir, f"{filename_base}_segment_{i}.pkl")
                
#                 # Create data dictionary in EEGDM format
#                 data_dict = {
#                     "data": segment_data.astype(np.float32),
#                     "label": labels.get(i, 0) if labels else 0
#                 }
                
#                 with open(output_filename, "wb") as f:
#                     pickle.dump(data_dict, f)
                
#                 output_files.append(output_filename)
            
#             self.logger.info(f"Saved {len(output_files)} TUEV segments")
        
#         return output_files
    
#     def process_directory(self, 
#                          input_dir: str, 
#                          output_dir: str,
#                          labels_dict: Optional[Dict] = None) -> List[str]:
#         """
#         Process all EDF files in a directory.
        
#         Args:
#             input_dir: Directory containing EDF files
#             output_dir: Directory to save processed files  
#             labels_dict: Optional dictionary mapping filenames to labels
            
#         Returns:
#             List of all output file paths
#         """
#         all_output_files = []
        
#         for filename in os.listdir(input_dir):
#             if filename.lower().endswith('.edf'):
#                 edf_path = os.path.join(input_dir, filename)
#                 file_labels = labels_dict.get(filename) if labels_dict else None
                
#                 output_files = self.process_edf_file(
#                     edf_path=edf_path,
#                     output_dir=output_dir,
#                     labels=file_labels
#                 )
#                 all_output_files.extend(output_files)
        
#         self.logger.info(f"Processed {len(all_output_files)} total segments")
#         return all_output_files


# def main():
#     """Example usage of the EEGDMPreprocessor."""
#     import argparse
    
#     parser = argparse.ArgumentParser(description="EEGDM-compatible EEG preprocessing")
#     parser.add_argument("input_path", help="Input EDF file or directory")
#     parser.add_argument("output_dir", help="Output directory")
#     parser.add_argument("--style", choices=['chbmit', 'tuev'], default='chbmit',
#                        help="Preprocessing style")
#     parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
#     args = parser.parse_args()
    
#     if args.verbose:
#         logging.basicConfig(level=logging.INFO)
    
#     # Initialize preprocessor
#     processor = EEGDMPreprocessor(style=args.style)
    
#     # Process input
#     if os.path.isfile(args.input_path):
#         # Single file
#         output_files = processor.process_edf_file(args.input_path, args.output_dir)
#         print(f"Processed 1 file → {len(output_files)} segments")
#     elif os.path.isdir(args.input_path):
#         # Directory
#         output_files = processor.process_directory(args.input_path, args.output_dir)
#         print(f"Processed directory → {len(output_files)} total segments")
#     else:
#         print(f"Error: {args.input_path} is not a valid file or directory")


# if __name__ == "__main__":
#     main()
