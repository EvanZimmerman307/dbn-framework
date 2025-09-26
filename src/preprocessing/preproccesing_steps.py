import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt
from preprocessing_base import register, PreprocessingStep, STEP_REGISTRY
from record import Record
from dataclasses import replace
from ecgdetectors import Detectors
import logging
import pandas as pd
from wfdb import processing

# @register("pan_tompkins")
# class PanTompkins(PreprocessingStep):
#     """Run the pan-tompkins algorithm to detect candidate R peaks"""
#     def __call__(self, record: Record) -> Record:
#         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#         logger = logging.getLogger(__name__)
#         x = record.ecg_recording.signal
#         try:
#             fs = int(self.params["fs"])
#             channels = self.params["channels"] # channels to try for PT
#         except Exception as e:
#             logger.info(f"sampling frequency (fs) and channel need to be specified in preprocessing config: {e}")
#         detectors = Detectors(fs)
        
#         # try the channels in the specified order
#         for channel in channels:
#             if channel in record.ecg_recording.channels: # if the channel is in the ecg recording channels, grab the index
#                 channel_index = record.ecg_recording.channels.index(channel)
#                 break
        
#         channel_signal = x[channel_index]
#         candidate_r_peaks = detectors.pan_tompkins_detector(channel_signal) #indices
#         candidate_r_peaks = np.asarray(candidate_r_peaks)
#         return replace(record, candidates=candidate_r_peaks)

@register("highpass_filter")
class HighpassFilter(PreprocessingStep):
    """Butterworth high-pass with zero-phase option; channel-wise."""
    def __call__(self, record: Record) -> Record:
        x = record.ecg_recording.signal
        fs = record.ecg_recording.fs
        cutoff = float(self.params.get("cutoff_hz", 0.5))
        order = int(self.params.get("order", 4))
        zero_phase = bool(self.params.get("zero_phase", True))

        if cutoff <= 0 or fs <= 0:
            return record  # no-op

        nyq = fs / 2.0
        wn = cutoff / nyq
        sos = butter(order, wn, btype="highpass", output="sos")

        if x.ndim == 1:
            x_f = sosfiltfilt(sos, x) if zero_phase else sosfilt(sos, x)
        else:
            x_f = np.vstack([
                sosfiltfilt(sos, ch) if zero_phase else sosfilt(sos, ch)
                for ch in x
            ])

        ecg_new = replace(record.ecg_recording, signal=x_f) # create a copy with specified field updated
        # audit trail
        stats = dict(record.ecg_recording.stats or {})
        stats.setdefault("cleaning", []).append({"op": "highpass_filter",
                                                 "cutoff_hz": cutoff,
                                                 "order": order,
                                                 "zero_phase": zero_phase})
        ecg_new = replace(ecg_new, stats=stats)
        return replace(record, ecg_recording=ecg_new)

@register("xqrs")
class XQRS(PreprocessingStep):
    """Run XQRS from wfdb for candidate detection"""
    def __call__(self, record: Record) -> Record:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        x = record.ecg_recording.signal
        try:
            fs = int(self.params["fs"])
            channels = self.params["channels"] # channels to try for PT
        except Exception as e:
            logger.info(f"sampling frequency (fs) and channel need to be specified in preprocessing config: {e}")

        # try the channels in the specified order
        for channel in channels:
            if channel in record.ecg_recording.channels: # if the channel is in the ecg recording channels, grab the index
                channel_index = record.ecg_recording.channels.index(channel)
                break
        
        channel_signal = x[channel_index]
        candidate_r_peaks = processing.xqrs_detect(sig=channel_signal, fs=fs) #indices
        candidate_r_peaks = np.asarray(candidate_r_peaks)
        return replace(record, candidates=candidate_r_peaks)

@register("normalize")
class NormalizeRobust(PreprocessingStep):
    """Per-record, per-channel robust z-score: (x - median) / MAD; optional clipping."""
    """We don't use mean and std because data has sharp spikes and noise bursts"""
    def __call__(self, record: Record) -> Record:
        x = record.ecg_recording.signal
        clip_sigma = self.params.get("clip_sigma", None)
        eps = 1e-9

        if x.ndim == 1:
            med = np.median(x, keepdims=True)
            mad = np.median(np.abs(x - med), keepdims=True)
            mad = np.maximum(mad, eps)
            xz = (x - med) / mad
            if clip_sigma is not None:
                xz = np.clip(xz, -float(clip_sigma), float(clip_sigma))
            med_list = [float(med)]
            scale_list = [float(mad)]
        else:
            med = np.median(x, axis=1, keepdims=True)                # (C,1)
            mad = np.median(np.abs(x - med), axis=1, keepdims=True)  # (C,1)
            mad = np.maximum(mad, eps)
            xz = (x - med) / mad
            if clip_sigma is not None:
                xz = np.clip(xz, -float(clip_sigma), float(clip_sigma))
            med_list = med.squeeze(axis=1).astype(float).tolist()
            scale_list = mad.squeeze(axis=1).astype(float).tolist()

        ecg_new = replace(record.ecg_recording, signal=xz)
        stats = dict(record.ecg_recording.stats or {})
        stats["norm"] = {
            "mode": "per_record_robust_mad",
            "median_by_ch": med_list,
            "scale_by_ch": scale_list,
            "clip_sigma": clip_sigma,
        }
        ecg_new = replace(ecg_new, stats=stats)
        return replace(record, ecg_recording=ecg_new)

@register("make_windows")
class Windowize(PreprocessingStep):
    """Make the table of ecg recording + candidate + annotation windows that will be consumed by downstream models"""
    def __call__(self, record: Record) -> Record:
        params = self.params
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        try:
            window_len = int(params["window_len"])
            stride = int(params["stride"])
            core = int(params["core"])
            # R_peak_symbols = params["R_peak_symbols"]
            # R_peak_symbols = set(R_peak_symbols)
            superclass_map = params["superclass_map"]
        except Exception as e:
            logger.info(f"window len, stride, core, and R peak symbols need to be specified in make_windows config: {e}")
        channels = record.ecg_recording.channels
        column_names = ['start', 'end', 'core_start', 'core_end'] + channels + ['annotation_symbols', 'annotation_ind', 'candidates']
        window_table = pd.DataFrame(columns=column_names)

        fs = record.ecg_recording.fs
        samples_per_window, samples_per_stride, samples_per_core = int(window_len * fs), int(stride * fs), int(core * fs)
        window_edge_count = (window_len - core) / 2 * fs # samples from window start to core

        signal = record.ecg_recording.signal # shape C x T
        num_samples = int(signal.shape[1])
        annotation_indices = record.annotation.indices
        annotation_symbols = record.annotation.symbols
        prev_annotation_i = 0
        prev_candidate_i = 0
        for i in range(0, num_samples, samples_per_stride):
            window_row = {}
            window_row['start'] = i
            window_row['end'] = i + samples_per_window - 1
            window_row['core_start'] = window_row['start'] + window_edge_count
            window_row['core_end'] = window_row['core_start'] + samples_per_core - 1
            for i, channel_name in enumerate(channels):
                channel_slice = signal[i][window_row['start']: window_row['end'] + 1]
                window_row[channel_name] = channel_slice
            
            if len(window_row[channels[0]]) < samples_per_window:
                continue  # Skip partial windows

            annotation_core = []
            annotation_ind = []
            i = prev_annotation_i
            while i < len(annotation_indices):
                ind, symbol = annotation_indices[i], annotation_symbols[i]
                if ind > window_row['core_end']:
                    break

                if ind >= window_row['core_start'] and symbol in superclass_map:
                    annotation_core.append(superclass_map[symbol])
                    annotation_ind.append(ind)
                
                i += 1 
            window_row['annotation_symbols'] = annotation_core
            window_row['annotation_ind'] = annotation_ind
            prev_annotation_i = i
            
            candidate_core = []
            i = prev_candidate_i
            while i < len(record.candidates):
                ind = record.candidates[i]
                if ind > window_row['core_end']:
                    break

                if ind >= window_row['core_start']:
                    candidate_core.append(ind)
                
                i += 1
            window_row['candidates'] = candidate_core
            prev_candidate_i = i
            
            next_ind = len(window_table)
            window_table.loc[next_ind] = window_row
            
        return replace(record, window_table=window_table)
                
                
            
            


        
