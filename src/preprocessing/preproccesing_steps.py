import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt
from preprocessing_base import register, PreprocessingStep, STEP_REGISTRY
from record import Record
from dataclasses import replace

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

@register("normalize_robust")
class NormalizeRobust(PreprocessingStep):
    """Per-record, per-channel robust z-score: (x - median) / MAD; optional clipping."""
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
