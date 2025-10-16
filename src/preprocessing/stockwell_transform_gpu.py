import torch
import numpy as np
import logging

def stockwell_transform_pytorch(signal, fmin, fmax, fs=1.0, device='cuda'):
    """
    PyTorch GPU-accelerated Stockwell (S) transform implementation.

    Parameters:
    - signal: 1D numpy array, the input signal
    - fmin: minimum frequency (in Hz, relative to sampling rate)
    - fmax: maximum frequency (in Hz, relative to sampling rate)  
    - fs: sampling frequency (default 1.0, frequencies are relative)
    - device: 'cuda' or 'cpu'

    Returns:
    - Complex numpy array of shape (n_freqs, n_times) containing the S-transform
    """
    logger = logging.getLogger(__name__)
    
    # Convert to torch tensor
    signal_tensor = torch.from_numpy(signal).float().to(device)
    N = len(signal_tensor)
    # logger.info(f"N: {N}")
    # logger.info(f"fs: {fs}")
    
    # Frequency range in samples
    fmin_samples = int(fmin * N / fs)
    # logger.info(f"fmin_samples: {fmin_samples}")
    fmax_samples = int(fmax * N / fs)
    # logger.info(f"fmax_samples: {fmax_samples}")
    
    # Ensure valid frequency range
    fmin_samples = max(1, fmin_samples)  # Avoid division by zero
    # logger.info(f"fmin_samples: {fmin_samples}")
    fmax_samples = min(N//2, fmax_samples)  # Nyquist limit
    # logger.info(f"fmax_samples: {fmax_samples}")
    
    n_freqs = fmax_samples - fmin_samples + 1
    # logger.info(f"Computing S-transform for {n_freqs} frequencies from {fmin_samples} to {fmax_samples}")
    
    # Pre-compute FFT of signal
    signal_fft = torch.fft.fft(signal_tensor)
    
    # Time vector
    t = torch.arange(N, dtype=torch.float32, device=device)
    
    # Initialize result tensor
    st_result = torch.zeros((n_freqs, N), dtype=torch.complex64, device=device)
    
    # For each frequency
    for i, f in enumerate(range(fmin_samples, fmax_samples + 1)):
        # Gaussian window width (sigma = 1/(2*pi*f))
        sigma = 1.0 / (2 * np.pi * f)
        
        # Create Gaussian window
        gaussian = torch.exp(-0.5 * ((t - N//2) / sigma)**2) / (sigma * torch.sqrt(torch.tensor(2 * np.pi, device=device)))
        
        # FFT of Gaussian
        gaussian_fft = torch.fft.fft(torch.fft.ifftshift(gaussian))
        
        # Convolution in frequency domain
        conv_result = torch.fft.ifft(signal_fft * gaussian_fft)
        
        # Phase shift for center time
        phase_shift = torch.exp(-1j * 2 * np.pi * f * t / N)
        st_result[i] = conv_result * phase_shift
    
    # Convert back to numpy
    return st_result.cpu().numpy()


def test_stockwell_pytorch():
    """Test function for the PyTorch Stockwell transform"""
    import time
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test signal
    t = np.linspace(0, 1, 1000, endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    
    print("Testing PyTorch Stockwell transform...")
    start_time = time.time()
    
    # Compute transform
    result = stockwell_transform_pytorch(signal, fmin=1, fmax=100, fs=1000, device=device)
    
    end_time = time.time()
    print(".3f")
    print(f"Result shape: {result.shape}")
    print(f"Result dtype: {result.dtype}")
    
    # Check for expected peaks
    power = np.abs(result)**2
    freq_peaks = np.argmax(np.mean(power, axis=1))
    print(f"Peak frequencies detected at indices: {freq_peaks}")
    
    return result


if __name__ == "__main__":
    test_stockwell_pytorch()
