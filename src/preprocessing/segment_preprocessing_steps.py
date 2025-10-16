import numpy as np
from preprocessing_base import register, PreprocessingStep, STEP_REGISTRY
from record import Record
from dataclasses import replace
import logging
import pandas as pd
from scipy import signal
# import neurokit2 as nk
# from stockwell import st
from stockwell_transform_gpu import stockwell_transform_pytorch
from collections import defaultdict
import scipy.sparse
import h5py
from pathlib import Path

@register("select_signal")
class SelectSignal(PreprocessingStep):
    """Select the signal to utilize for classifying Arrhythmia"""
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        x = record.ecg_recording.signal
        try:
            candidate_channels = self.params["candidate_channels"]
        except Exception as e:
            logger.info(f"Need to specify which channel to use (in order of preference): {e}")
        
        selected_signal = None
        chosen_channel = None
        for channel in candidate_channels:
            if channel in record.ecg_recording.channels:
                channel_index = record.ecg_recording.channels.index(channel)
                selected_signal = x[channel_index]
                chosen_channel = channel
                logger.info(f"Using channel: {chosen_channel}")
                break
        
        if selected_signal is None:
            raise ValueError("Couldn't find a signal that matches the channels specified in params")

        ecg_new = replace(record.ecg_recording, channels=[chosen_channel], signal=selected_signal)
        return replace(record, ecg_recording=ecg_new)

@register("low_pass_filter")
class LowPassFilter(PreprocessingStep):
    """
    Apply a Butterworth low-pass filter to the data.
    A low-pass filter lets the slow stuff (low frequencies) through and removes the fast stuff (high frequencies).
    An ECG signal has slow parts (the P, QRS, T waves — which are important) and fast parts (muscle noise, electrical interference). 
    
    Parameters:
    - cutoff_freq: cutoff frequency in Hz
    - order: filter order (higher = sharper cutoff, default=5)
    """
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        try:
            cutoff_freq = self.params["cutoff_freq"]
            order = self.params["order"]
        except Exception as e:
            logger.info(f"Need to specify cutoff_freq and order: {e}")
        
        nyquist = record.ecg_recording.fs / 2
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = signal.filtfilt(b, a, record.ecg_recording.signal)
        
        ecg_new = replace(record.ecg_recording, signal=filtered_signal)
        return replace(record, ecg_recording=ecg_new)

@register("downsample")
class Downsample(PreprocessingStep):
    """
    Downsample the signal
    
    Parameters:
    - target_fs: The target downsampling frequency
    """
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        try:
            target_fs = self.params["target_fs"]
        except Exception as e:
            logger.info(f"Need to specify target_fs for downsampling: {e}")
        
        # Calculate the number of samples in the new signal
        total_samples = len(record.ecg_recording.signal)
        total_seconds = total_samples / record.ecg_recording.fs
        target_samples = total_seconds * target_fs
        signal_downsampled = signal.resample(record.ecg_recording.signal, int(target_samples))

        ecg_new = replace(record.ecg_recording, signal=signal_downsampled, fs=target_fs)
        return replace(record, ecg_recording=ecg_new)

    
@register("windowize")
class Windowize(PreprocessingStep):
    """
    Cut the signal into windows of a specified duration and normalize
    (x - median) / IQR
    stable and morphology preserving
    """
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
        try: 
            window_duration = self.params["window_duration"] # in seconds
        except Exception as e:
            logger.info(f"Need to specify window duration in seconds")
        
        window_samples = int(window_duration * record.ecg_recording.fs)
        windowed_record = []
        for i in range(0, len(record.ecg_recording.signal), window_samples):
            new_window = record.ecg_recording.signal[i:i+window_samples]
            window_median = np.median(new_window)
            iqr = np.percentile(new_window, 75) - np.percentile(new_window, 25)
            new_window = (new_window - window_median) / (iqr if iqr != 0 else 1)

            if len(new_window) < window_duration * record.ecg_recording.fs:
                continue

            windowed_record.append(new_window)

        
        ecg_new = replace(record.ecg_recording, signal=np.array(windowed_record))
        return replace(record, ecg_recording=ecg_new)

@register("detrend")
class Detrend(PreprocessingStep):
    """
    ECG signals often “wander” up and down over time, not because the heart changed, 
    but because of things like electrode movement, breathing, or baseline drift. Detrending removes that slow, wavy drift, 
    so the baseline of the ECG stays flat and the actual heartbeats stand out clearly.
    """
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        try:
            regularization_param = self.params["regularization"]
        except Exception as e:
            logger.info(f"Need to specify regularization: {e}")

        detrended_signal = []
        # Outside loop - compute once for all windows of same length
        N = len(record.ecg_recording.signal[0])  # window length
        identity = np.eye(N)
        D_2 = _build_D2_matrix(N)
        inv_cached = np.linalg.inv(identity + regularization_param**2 * D_2.T @ D_2)
        detrend_matrix = identity - inv_cached  # Pre-compute

        for signal in record.ecg_recording.signal:
            ecg_detrended = detrend_matrix @ signal
            detrended_signal.append(ecg_detrended)

        ecg_new = replace(record.ecg_recording, signal=np.array(detrended_signal))
        return replace(record, ecg_recording=ecg_new)
    

@register("stockwell_transform")
class StockwellTransform(PreprocessingStep):
    """
    Apply the stockwell transform to the signal
    """
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        try:
            fmin = self.params["fmin"]
            fmax = self.params["fmax"]
            signal_length = self.params["signal_length"]
        except Exception as e:
            logger.info(f"Need to specify fmin, fmax, and signal_length in config params")
        
        # df = 1 / signal_length
        # fmin_samples = int(fmin / df)
        # fmax_samples = int(fmax / df)
        
        transformed_signals = []

        for i in range(len(record.ecg_recording.signal)): 
            # logger.info(f"Stockwell params: fmin={fmin}, fmax={fmax}, signal_length={signal_length}")
            # logger.info(f"Calculated: df={df}, fmin_samples={fmin_samples}, fmax_samples={fmax_samples}")
            # logger.info(f"Signal shape: {record.ecg_recording.signal[i].shape}")
            trans_signal = stockwell_transform_pytorch(record.ecg_recording.signal[i].squeeze(), fmin, fmax, fs=record.ecg_recording.fs)
            real_part = np.real(trans_signal)
            imag_part = np.imag(trans_signal)
            transformed_signal = np.stack((real_part, imag_part), axis=0)
            transformed_signals.append(transformed_signal)
            
        ecg_new = replace(record.ecg_recording, signal=transformed_signals)
        return replace(record, ecg_recording=ecg_new)

@register("make_dataset")
class MakeDataset(PreprocessingStep):
    """
    Make the window table
    """
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        try:
            original_fs = self.params["original_fs"]
            superclass_map = self.params["superclass_map"]
            signal_length = self.params["signal_length"]
            output_path = self.params['output_path']
        except Exception as e:
            logger.info(f"Need to specify original_fs and superclass_map in config params")
        
        column_names = ['start_i', 'end_i'] + record.ecg_recording.channels + ['label']
        window_table = pd.DataFrame(columns=column_names)
        fs = record.ecg_recording.fs
        
        label_i = 0
        annotation_indices = record.annotation.indices
        annotation_symbols = record.annotation.symbols

        signals_list = []
        labels_list = []
        record_ids_list = []
        start_i_list = []
        end_i_list = []
        channel_list = []
        for i in range(len(record.ecg_recording.signal)):
            start = i * record.ecg_recording.signal[i].shape[-1]
            start_t = start / fs # in seconds
            end = start + record.ecg_recording.signal[i].shape[-1] - 1
            end_t = end / fs # in seconds
            sig = record.ecg_recording.signal[i]

            # Don't add partial signals
            if sig.shape[-1] < signal_length * fs:
                continue

            label_count = defaultdict(int)
            label_i_time = annotation_indices[label_i] / original_fs
            while label_i_time < end_t and label_i < len(annotation_indices):
                if label_i_time > start_t:
                    symbol = annotation_symbols[label_i]
                    if symbol in superclass_map:
                        superclass_symbol = superclass_map[symbol]
                        label_count[superclass_symbol] += 1
                
                # otherwise this annotation is before start_t
                label_i += 1
                if label_i >= len(annotation_indices):
                    break
                
                label_i_time = annotation_indices[label_i] / original_fs
            
            # All rows need labels
            if len(label_count) == 0:
                continue

            majority_superclass_symbol = max(label_count.items(), key=lambda x: x[1])[0]
            
            start_i_list.append(start)
            end_i_list.append(end)
            channel_list.append(record.ecg_recording.channels[0])
            signals_list.append(sig)
            labels_list.append(majority_superclass_symbol)
            record_ids_list.append(record.ecg_recording.record_id)

        # Convert to numpy arrays
        signals_array = np.array(signals_list)  # Shape: (n_windows, 2, 150, 1000)
        labels_array = np.array(labels_list, dtype='S10')  # Unicode strings
        record_ids_array = np.array(record_ids_list, dtype='S10')
        start_i_array = np.array(start_i_list)
        end_i_array = np.array(end_i_list)
        channel_array = np.array(channel_list, dtype='S10')

        # Save to HDF5
        output_path = Path(f"{output_path}/{record.split_type}")
        output_path.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        file_path = output_path / f"{record.ecg_recording.record_id}.h5"  # Use record.record_id
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('signals', data=signals_array, compression='gzip')
            f.create_dataset('labels', data=labels_array)
            f.create_dataset('record_ids', data=record_ids_array)
            f.create_dataset('start_i', data=start_i_array)
            f.create_dataset('end_i', data=end_i_array)
            f.create_dataset('channel_name', data=channel_array) 
           
        logger.info(f"Wrote preprocessed data for record {record.ecg_recording.record_id} to preprocessing/{record.split_type}")
        return None

def _build_D2_matrix(N):
    """Build the second-order difference matrix for Tarvainen detrending."""
    B = np.dot(np.ones((N - 2, 1)), np.array([[1, -2, 1]]))
    D_2 = scipy.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 2, N))
    return D_2

        


 

    