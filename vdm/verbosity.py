"""
Verbosity control for VDM-BIND.

Provides a consistent verbosity system across the codebase with three levels:
- SILENT (0): No output at all - ideal for production/batch jobs
- SUMMARY (1): Minimal output - model name, checkpoint path, key stats only
- DEBUG (2): Full verbose output - useful for debugging

Usage
-----
Global setting:
    >>> import vdm
    >>> vdm.set_verbosity('silent')  # or 0, 1, 2
    >>> vdm.set_verbosity('summary')
    >>> vdm.set_verbosity('debug')

Context manager for temporary silence:
    >>> with vdm.quiet():
    ...     # All operations are silent
    ...     samples = sample(model, cond)

Check current level:
    >>> vdm.get_verbosity()
    1
"""

from contextlib import contextmanager
from typing import Union
import sys

# Verbosity levels
SILENT = 0
SUMMARY = 1
DEBUG = 2

# String aliases for convenience
_LEVEL_ALIASES = {
    'silent': SILENT,
    'quiet': SILENT,
    'summary': SUMMARY,
    'normal': SUMMARY,
    'debug': DEBUG,
    'verbose': DEBUG,
}

# Global verbosity state
_verbosity_level = SUMMARY  # Default to summary


def set_verbosity(level: Union[int, str]) -> None:
    """
    Set the global verbosity level.
    
    Parameters
    ----------
    level : int or str
        Verbosity level. Can be:
        - Integer: 0 (silent), 1 (summary), 2 (debug)
        - String: 'silent'/'quiet', 'summary'/'normal', 'debug'/'verbose'
    
    Examples
    --------
    >>> import vdm
    >>> vdm.set_verbosity('silent')  # No output
    >>> vdm.set_verbosity(1)         # Summary only
    >>> vdm.set_verbosity('debug')   # Full output
    """
    global _verbosity_level
    
    if isinstance(level, str):
        level_lower = level.lower()
        if level_lower not in _LEVEL_ALIASES:
            valid = list(_LEVEL_ALIASES.keys())
            raise ValueError(f"Unknown verbosity level '{level}'. Valid options: {valid}")
        _verbosity_level = _LEVEL_ALIASES[level_lower]
    elif isinstance(level, (int, float)):
        _verbosity_level = int(level)
    else:
        raise TypeError(f"Verbosity level must be int or str, got {type(level)}")


def get_verbosity() -> int:
    """
    Get the current global verbosity level.
    
    Returns
    -------
    int
        Current verbosity level (0=silent, 1=summary, 2=debug)
    """
    return _verbosity_level


def vprint(*args, level: int = SUMMARY, **kwargs) -> None:
    """
    Print if current verbosity level is >= specified level.
    
    Parameters
    ----------
    *args : Any
        Arguments to pass to print()
    level : int, optional
        Minimum verbosity level required to print (default: SUMMARY)
    **kwargs : Any
        Keyword arguments to pass to print()
    
    Examples
    --------
    >>> vprint("Always shown at summary+", level=SUMMARY)
    >>> vprint("Only shown in debug mode", level=DEBUG)
    >>> vprint("Model loaded", level=SUMMARY)
    """
    if _verbosity_level >= level:
        print(*args, **kwargs)


def vprint_summary(*args, **kwargs) -> None:
    """Print at SUMMARY level (shown unless silent)."""
    vprint(*args, level=SUMMARY, **kwargs)


def vprint_debug(*args, **kwargs) -> None:
    """Print at DEBUG level (only shown in debug mode)."""
    vprint(*args, level=DEBUG, **kwargs)


def is_silent() -> bool:
    """Check if verbosity is set to silent."""
    return _verbosity_level == SILENT


def is_debug() -> bool:
    """Check if verbosity is set to debug."""
    return _verbosity_level >= DEBUG


@contextmanager
def quiet():
    """
    Context manager for temporarily silencing all output.
    
    Yields
    ------
    None
    
    Examples
    --------
    >>> with vdm.quiet():
    ...     # All vprint calls are suppressed
    ...     model = load_model()
    ...     samples = generate(model, cond)
    >>> # Normal verbosity restored
    """
    global _verbosity_level
    old_level = _verbosity_level
    _verbosity_level = SILENT
    try:
        yield
    finally:
        _verbosity_level = old_level


@contextmanager
def verbosity(level: Union[int, str]):
    """
    Context manager for temporarily changing verbosity level.
    
    Parameters
    ----------
    level : int or str
        Temporary verbosity level
    
    Yields
    ------
    None
    
    Examples
    --------
    >>> with vdm.verbosity('debug'):
    ...     # Full debug output
    ...     model = load_model()
    >>> # Original verbosity restored
    """
    global _verbosity_level
    old_level = _verbosity_level
    set_verbosity(level)
    try:
        yield
    finally:
        _verbosity_level = old_level


def verbose_to_level(verbose: bool) -> int:
    """
    Convert legacy verbose=True/False to verbosity level.
    
    For backward compatibility with existing code that uses verbose parameter.
    
    Parameters
    ----------
    verbose : bool
        Legacy verbose flag
    
    Returns
    -------
    int
        Verbosity level (DEBUG if True, SILENT if False)
    """
    return DEBUG if verbose else SILENT


class VerbosityContext:
    """
    Class-based verbosity context for methods that need to track verbosity.
    
    Can be used as a mixin or standalone helper.
    
    Parameters
    ----------
    verbose : bool or int or str, optional
        Verbosity setting. If None, uses global setting.
        If bool, converts to level (True=DEBUG, False=SILENT).
        If int or str, uses that level directly.
    
    Examples
    --------
    >>> ctx = VerbosityContext(verbose=True)
    >>> ctx.vprint("Debug message")  # Only prints if verbose
    """
    
    def __init__(self, verbose: Union[bool, int, str, None] = None):
        if verbose is None:
            self._local_level = None  # Use global
        elif isinstance(verbose, bool):
            self._local_level = verbose_to_level(verbose)
        elif isinstance(verbose, str):
            self._local_level = _LEVEL_ALIASES.get(verbose.lower(), SUMMARY)
        else:
            self._local_level = int(verbose)
    
    @property
    def verbosity_level(self) -> int:
        """Get effective verbosity level (local or global)."""
        if self._local_level is not None:
            return self._local_level
        return _verbosity_level
    
    def vprint(self, *args, level: int = SUMMARY, **kwargs) -> None:
        """Print if verbosity level is sufficient."""
        if self.verbosity_level >= level:
            print(*args, **kwargs)
    
    def vprint_summary(self, *args, **kwargs) -> None:
        """Print at SUMMARY level."""
        self.vprint(*args, level=SUMMARY, **kwargs)
    
    def vprint_debug(self, *args, **kwargs) -> None:
        """Print at DEBUG level."""
        self.vprint(*args, level=DEBUG, **kwargs)
    
    def is_silent(self) -> bool:
        """Check if effectively silent."""
        return self.verbosity_level == SILENT
    
    def is_debug(self) -> bool:
        """Check if in debug mode."""
        return self.verbosity_level >= DEBUG
