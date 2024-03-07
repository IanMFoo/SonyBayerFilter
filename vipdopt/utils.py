"""Module containing common functions used throughout the package."""

import contextlib
import functools
import importlib.util as imp
import itertools
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Generator, Iterable
from importlib.abc import Loader
from multiprocessing.pool import AsyncResult
from numbers import Number
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, TypeVar, TypedDict

import numpy as np
import numpy.typing as npt
import yaml
from yaml.constructor import SafeConstructor

PathLike = TypeVar('PathLike', str, Path, bytes, os.PathLike)

# Generic types for type hints
T = TypeVar('T')
R = TypeVar('R')

P = ParamSpec('P')
Q = ParamSpec('Q')


class TruncateFormatter(logging.Formatter):
    """Logging formatter for truncating large output."""
    def __init__(
            self,
            max_length: int=300,
            log_file: str='dev.log',
            level: int=logging.WARNING,
            **kwargs,
        ):
        """Initialize a TruncateFormatter."""
        super().__init__(**kwargs)
        self.max_length = max_length
        self.log_file = log_file
        self.level = level

    def format(self, record):  # noqa: A003
        """Format the record. Truncates long messages if not DEBUG level."""
        msg = super().format(record)
        if len(msg) > self.max_length and self.level > logging.DEBUG:
            return f"""{msg[:self.max_length]}...\nOutput truncated.
To see full output, run with -vv or check {self.log_file}\n"""
        return msg


def setup_logger(
        name: str,
        level: int=logging.INFO,
        log_file: str='dev.log',
    ) -> logging.Logger | logging.LoggerAdapter:
    """Setup logger to use across the program."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


    # Handle stream output by truncating long messages.
    # Only prints messages with no additional info.
    shandler = logging.StreamHandler()
    shandler.setFormatter(TruncateFormatter(
        fmt='%(message)s',
        log_file=log_file,
        level=level,
    ))
    shandler.setLevel(level)

    # Create a handler for putting info into a .log file. Includes time stamps etc.
    # Will write EVERYTHING to the log (i.e. level = debug)
    if not os.path.exists(os.path.dirname(log_file)):
        with contextlib.suppress(Exception):
            os.makedirs(os.path.dirname(log_file))

    fhandler = logging.FileHandler(log_file, mode='a')
    fhandler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    logger.addHandler(shandler)
    logger.addHandler(fhandler)

    return logger


def sech(z: npt.ArrayLike | Number) -> npt.ArrayLike | Number:
    """Hyperbolic Secant."""
    return 1.0 / np.cosh(np.asanyarray(z))

def import_lumapi(loc: str):
    """Import the Lumerical python api from a specified location."""
    lumapi = object()
    try:
        spec_lin = imp.spec_from_file_location('lumapi', loc)

        # These assertions are so MyPy doesn't get mad
        assert spec_lin is not None
        assert isinstance(spec_lin.loader, Loader)

        lumapi = imp.module_from_spec(spec_lin)
        spec_lin.loader.exec_module(lumapi)
    except FileNotFoundError:
        logging.exception('lumapi not found. Using dummy values\n')
    return lumapi


def convert_path(path: PathLike) -> Path:
    """Ensure that a Path is a Path object."""
    if isinstance(path, Path):
        return path
    if isinstance(path, bytes):
        return Path(path.decode())
    if isinstance(path, str | os.PathLike):
        return Path(path)

    # Input was not a PathLike
    raise ValueError(f'Argument must be PathLike, got {type(path)}')


def ensure_path(
        func: Callable[Concatenate[Any, Path, P], R],
) -> Callable[Concatenate[Any, PathLike, P], R]:
    """Function decorator for converting PathLike's to Path's."""
    @functools.wraps(func)
    def wrapper(
            arg0: Any,
            path: PathLike,
            *args: P.args,
            **kwargs: P.kwargs,
    ) -> R:
        return func(arg0, convert_path(path), *args, **kwargs)
    return wrapper


def save_config_file(
        config_data: dict,
        fname: PathLike,
        cfg_format: str='auto',
        **kwargs,
):
    """Save a configuration file."""
    path_filename = convert_path(fname)
    if cfg_format == 'auto':
        cfg_format = path_filename.suffix

    if type(config_data) is not dict:
        config_data = config_data.data     #! 20240221: Hacked together by Ian to make this work, JSON wasn't dumping file of type Config properly

    match cfg_format.lower():
        case '.yaml' | '.yml':
            with path_filename.open('w') as f:
                yaml.dump(config_data, f, **kwargs)
        case '.json':
            with path_filename.open('w') as f:
                json.dump(
                    config_data,
                    f,
                    indent=4,
                    ensure_ascii=True,
                    **kwargs
                )
        case _:
            msg = f'{cfg_format} file saving not yet supported.'
            raise NotImplementedError(msg)


def read_config_file(fname: PathLike, cfg_format: str='auto') -> dict[str, Any] | None:
    """Read a configuration file."""
    path_filename = convert_path(fname)
    if cfg_format == 'auto':
        cfg_format = path_filename.suffix

    config_loader = _get_config_loader(cfg_format)
    return config_loader(path_filename)


def _get_config_loader(cfg_format: str) -> Callable[[PathLike], dict]:
    """Return a configuration file loader depending on the format."""
    match cfg_format.lower():
        case '.yaml' | '.yml':
            return _yaml_loader
        case '.json':
            return _json_loader
        case _:
            msg = f'{cfg_format} file loading not yet supported.'
            raise NotImplementedError(msg)


def _json_loader(fname: PathLike) -> dict:
    with open(fname) as stream:
        return json.load(stream)


def _yaml_loader(fname: PathLike) -> dict:
    """Config file loader for YAML files."""
    # Allow the safeloader to convert sequences to tuples
    SafeConstructor.add_constructor(  # type: ignore
        'tag:yaml.org,2002:python/tuple',
        lambda self, x: tuple(SafeConstructor.construct_sequence(self, x))
    )
    with open(fname, 'rb') as stream:
        return yaml.safe_load(stream)

def subclasses(cls: T) -> list[type[T]]:
    """Get all subclasses of a given class."""
    return [scls.__name__ for scls in cls.__subclasses__()]


def wait_for_results(results: list[AsyncResult]):
    while True:
        time.sleep(1)
        # catch exception if results are not ready yet
        try:
            ready = [result.ready() for result in results]
            successful = [result.successful() for result in results]
        except Exception:
            continue
        # exit loop if all tasks returned success
        if all(successful):
            break
        # raise exception reporting exceptions received from workers
        if all(ready) and not all(successful):
            raise Exception(f'Workers raised following exceptions {[result._value for result in results if not result.successful()]}')


def ladd_to_all(l: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the left."""
    for item in l:
        yield val + item

def radd_to_all(l: Iterable[T], val: T) -> Generator[T, None, None]:
    """Add a value to all memebers of a list on the right."""
    for item in l:
        yield item + val

def split_glob(pattern: str) -> Generator[str, None, None]:
    """Convert a glob pattern into multiple patterns if necessary."""
    if '{' in pattern:
        start = pattern.index('{')
        end = pattern.index('}')

        prefix = pattern[:start]
        choices = pattern[start + 1:end].split(',')
        for choice in choices:
            yield from ladd_to_all(split_glob(pattern[end + 1:]), prefix + choice.strip())
    else:
        yield pattern

def glob_first(search_dir: PathLike, pattern: str) -> Path | None:
    """Find the first file to match the given pattern."""
    patterns = list(split_glob(pattern))
    search_path = convert_path(search_dir)
    matches = itertools.chain.from_iterable(map(search_path.glob, patterns))
    try:
        first_match = next(matches)
    except StopIteration as e:
        msg = f'No path found with pattern {pattern} in {search_path}'
        raise FileNotFoundError(msg) from e
    return first_match


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.
    """

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class Coordinates(TypedDict):
    x: npt.NDArray
    y: npt.NDArray
    z: npt.NDArray