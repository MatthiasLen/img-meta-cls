import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
import os
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

def plot_batch_per_sample(batch, figsize=(15, 10), title = None, id = 1):
    """
    Plots each sample (set of images) in a row.

    Args:
        batch (Tensor): Tensor of shape (B, N, 1, H, W)
        figsize (tuple): Size of the figure.
        title (str): Optional title.
    """
    if not os.path.isdir("./tmp"):
        os.makedirs("./tmp")
    
    B, N, C, H, W = batch.shape
    fig, axes = plt.subplots(B, N, figsize=figsize)

    if title:
        fig.suptitle(title, fontsize=16)

    for i in range(B):
        # Compute stats for sample i (all its N images)
        sample_pixels = batch[i].view(-1)
        stats_text = (
            f"min: {sample_pixels.min().item():.3f}\n"
            f"max: {sample_pixels.max().item():.3f}\n"
            f"mean: {sample_pixels.mean().item():.3f}\n"
            f"std: {sample_pixels.std().item():.3f}\n"
            f"median: {sample_pixels.median().item():.3f}"
        )

        # Plot images
        for j in range(N):
            img = batch[i, j].squeeze().cpu().numpy()
            ax = axes[i][j] if B > 1 else axes[j]
            ax.imshow(img, cmap='gray')
            ax.axis('off')

        # Add stats text to left of the row (using the first image axis)
        # Adjust position to the left outside the image
        ax_stats = axes[i][0] if B > 1 else axes[0]
        ax_stats.text(
            -0.5, 0.5, stats_text,
            fontsize=10,
            va='center', ha='right',
            transform=ax_stats.transAxes,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.savefig(f"./tmp/{id}.png", dpi=300, bbox_inches='tight')
    plt.close()


def normalize_per_sample(batch):
    """
    Normalize each sample in the batch independently with z-score:
    - Compute mean and std over all voxels of that sample.
    - Supports 3D (B, N, H, W), 4D (B, C, H, W), and 5D (B, C, D, H, W).
    """
    B = batch.shape[0]
    if batch.ndim == 5:
        mean = batch.view(B, -1).mean(dim=1).view(B, 1, 1, 1, 1)
        std = batch.view(B, -1).std(dim=1).view(B, 1, 1, 1, 1)
    elif batch.ndim == 4:
        mean = batch.view(B, -1).mean(dim=1).view(B, 1, 1, 1)
        std = batch.view(B, -1).std(dim=1).view(B, 1, 1, 1)
    elif batch.ndim == 3:
        mean = batch.view(B, -1).mean(dim=1).view(B, 1, 1)
        std = batch.view(B, -1).std(dim=1).view(B, 1, 1)
    else:
        raise ValueError("Input batch must be 3D, 4D or 5D tensor")
    std = std.clamp(min=1e-8)
    return (batch - mean) / std

def count_parameters(model):
    """
    Returns the total number and tranable number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Total number of parameters.
        int: Number of trainable parameters
    """
    total_p =  sum(p.numel() for p in model.parameters())
    trainable_p =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_p, trainable_p


def generate_timestamped_log_path(base_path="./logs/training.log",
                                 timestamp_format="%Y%m%d_%H%M%S"):
    """
    Generate a timestamped log file path to distinguish between experiments.

    Args:
        base_path (str): Base log file path (e.g., "./logs/training.log")
        timestamp_format (str): Format string for timestamp (default: YYYYMMDD_HHMMSS)

    Returns:
        str: Timestamped log file path

    Example:
        Input: "./logs/training.log"
        Output: "./logs/training_20241024_143052.log"
    """
    # Split the path into directory, name, and extension
    log_dir = os.path.dirname(base_path)
    filename = os.path.basename(base_path)

    # Split filename into name and extension
    if '.' in filename:
        name, ext = filename.rsplit('.', 1)
        ext = '.' + ext
    else:
        name, ext = filename, ''

    # Generate timestamp
    timestamp = datetime.now().strftime(timestamp_format)

    # Create timestamped filename
    timestamped_filename = f"{name}_{timestamp}{ext}"

    # Combine back into full path
    if log_dir:
        return os.path.join(log_dir, timestamped_filename)
    else:
        return timestamped_filename


def setup_console_logging(log_file_path="./logs/training.log",
                         log_level=logging.INFO,
                         max_bytes=10*1024*1024,  # 10MB
                         backup_count=5,
                         console_level=logging.INFO,
                         use_timestamp=True,
                         timestamp_format="%Y%m%d_%H%M%S"):
    """
    Set up logging to capture both console output and log to file with rotation.

    Args:
        log_file_path (str): Path to the log file
        log_level (int): Logging level for file output
        max_bytes (int): Maximum size of each log file before rotation
        backup_count (int): Number of backup log files to keep
        console_level (int): Logging level for console output
        use_timestamp (bool): Whether to add timestamp to filename (default: True)
        timestamp_format (str): Format for timestamp if use_timestamp=True

    Returns:
        tuple: (logging.Logger, str) - Logger instance and actual log file path used
    """
    # Generate timestamped filename if requested
    if use_timestamp:
        actual_log_path = generate_timestamped_log_path(log_file_path, timestamp_format)
    else:
        actual_log_path = log_file_path

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(actual_log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger('IMC')
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Prevent propagation to the root logger to avoid duplicate logs
    logger.propagate = False

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        actual_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Log initial setup message
    logger.info(f"Logging initialized - File: {actual_log_path}, Level: {logging.getLevelName(log_level)}")

    return logger, actual_log_path


def setup_params_logging(log_dir="./logs",
                           max_bytes=10*1024*1024,
                           backup_count=5):
    """
    Convenience function to set up logging for experiments with automatic timestamping.

    Args:
        experiment_name (str): Name of the experiment (becomes part of filename)
        log_dir (str): Directory to store log files
        log_level (int): Logging level for file output
        max_bytes (int): Maximum size of each log file before rotation
        backup_count (int): Number of backup log files to keep

    Returns:
        tuple: (logging.Logger, str) - Logger instance and actual log file path

    Example:
        logger, log_path = setup_params_logging("resnet_training")
        # Creates: ./logs/resnet_training_20241024_143052.log
    """
    log_file_path = os.path.join(log_dir, f"params.log")
    logger = logging.getLogger('IMC_params')
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.info(f"Parameter logging initialized - File: {log_file_path}, Level: {logging.getLevelName(logging.DEBUG)}")
    return logger

def setup_experiment_logging(experiment_name="training",
                           log_dir="./logs",
                           log_level=logging.INFO,
                           max_bytes=10*1024*1024,
                           backup_count=5,
                           console_level=logging.INFO):
    """
    Convenience function to set up logging for experiments with automatic timestamping.

    Args:
        experiment_name (str): Name of the experiment (becomes part of filename)
        log_dir (str): Directory to store log files
        log_level (int): Logging level for file output
        max_bytes (int): Maximum size of each log file before rotation
        backup_count (int): Number of backup log files to keep
        console_level (int): Logging level for console output

    Returns:
        tuple: (logging.Logger, str) - Logger instance and actual log file path

    Example:
        logger, log_path = setup_experiment_logging("resnet_training")
        # Creates: ./logs/resnet_training_20241024_143052.log
    """
    log_file_path = os.path.join(log_dir, f"{experiment_name}.log")
    return setup_console_logging(
        log_file_path=log_file_path,
        log_level=log_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        console_level=console_level,
        use_timestamp=True
    )


class TeeOutput:
    """
    A class to duplicate stdout/stderr to both console and log file.
    """
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ""
        self._is_writing = False

    def write(self, message):
        # Write to original output
        if hasattr(self, 'original'):
            self.original.write(message)
            self.original.flush()

        if self._is_writing:
            return

        try:
            self._is_writing = True
            # Buffer the message and log complete lines
            self.buffer += message
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line.strip():  # Only log non-empty lines
                    self.logger.log(self.level, line.strip())
        finally:
            self._is_writing = False

    def flush(self):
        if hasattr(self, 'original'):
            self.original.flush()


@contextmanager
def capture_console_to_log(logger, capture_stdout=True, capture_stderr=True):
    """
    Context manager to capture stdout/stderr to log file while preserving console output.

    Args:
        logger: Logger instance to write to
        capture_stdout (bool): Whether to capture stdout
        capture_stderr (bool): Whether to capture stderr

    Usage:
        logger = setup_console_logging()
        with capture_console_to_log(logger):
            print("This will appear in both console and log file")
    """
    # Store original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Find and temporarily remove the console handler to prevent duplication
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
            console_handler = handler
            logger.removeHandler(handler)
            break

    # Create tee objects
    stdout_tee = None
    stderr_tee = None

    try:
        if capture_stdout:
            stdout_tee = TeeOutput(logger, logging.INFO)
            stdout_tee.original = original_stdout
            sys.stdout = stdout_tee

        if capture_stderr:
            stderr_tee = TeeOutput(logger, logging.ERROR)
            stderr_tee.original = original_stderr
            sys.stderr = stderr_tee

        yield logger

    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Add the console handler back if it was removed
        if console_handler and console_handler not in logger.handlers:
            logger.addHandler(console_handler)


def log_training_start(logger, model=None, config=None):
    """
    Log training session start information.

    Args:
        logger: Logger instance
        model: PyTorch model (optional)
        config: Configuration dictionary (optional)
    """
    logger.info("="*80)
    logger.info(f"TRAINING SESSION STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    if model is not None:
        total_params, trainable_params = count_parameters(model)
        logger.info(f"Model Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")

    if config is not None:
        logger.info("Configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

    logger.info("-"*80)


def log_training_end(logger, final_metrics=None):
    """
    Log training session end information.

    Args:
        logger: Logger instance
        final_metrics: Dictionary of final training metrics (optional)
    """
    logger.info("-"*80)
    logger.info(f"TRAINING SESSION COMPLETED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if final_metrics is not None:
        logger.info("Final Metrics:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value}")

    logger.info("="*80)


