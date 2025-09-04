from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass(kw_only=True)
class EcgRecording:
    signal: np.ndarray #  shape: (C, T) or (T,) for single-channel
    fs: float
    channels: list[str]
    record_id: str
    stats: Dict[str, Any] = None

@dataclass(kw_only=True)
class Annotation:
    annotation_indices: np.ndarray
    annotation_symbols: list[str]


@dataclass(kw_only=True) 
class Record:
    ecg_recording: EcgRecording
    annotation: Annotation


    @classmethod
    def from_wfdb(cls, recording, annotation, record_id):
        """A factory method to create an ECG record dataclass from a wfdb record"""
        record_params = {}

        ecg_recording_params = {}
        ecg_recording_params["signal"] = np.asarray(recording.p_signal).T if recording.p_signal.ndim == 2 else np.asarray(recording.p_signal)
        ecg_recording_params["fs"] = float(recording.fs)
        ecg_recording_params["channels"] = recording.sig_name
        ecg_recording_params["record_id"] = str(record_id)
        record_params["ecg_recording"] = EcgRecording(**ecg_recording_params)

        annotation_params = {}
        annotation_params["annotation_indices"] = np.asarray(annotation.sample, dtype=int)
        annotation_params["annotation_symbols"] = annotation.symbol
        record_params["annotation"] = Annotation(**annotation_params)
        
        return Record(**record_params)



