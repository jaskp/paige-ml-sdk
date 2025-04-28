import logging
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Union, List

import torch
import pandas as pd
from torch import Tensor

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


def load_torch(p: Union[PathLike, Iterable[PathLike]]) -> Tensor:
    """loads embeddings using torch.load"""
    if isinstance(p, (str, Path)):
        return torch.load(p)['embeddings']  # type: ignore[no-any-return]
    else:
        embeddings = []
        for p_ in p:
            embedder_output = torch.load(p_)
            embeddings.append(embedder_output['embeddings'])
        return torch.cat(embeddings)


def load_parquet(
    p: Union[PathLike, Iterable[PathLike]], feature_columns: Optional[List[str]] = None
) -> Tensor:
    """loads embeddings from parquet files containing feature columns

    Args:
        p: Path to parquet file or files
        feature_columns: List of feature column names. If None, columns matching 'feature_*' will be used

    Returns:
        Tensor containing embeddings
    """
    if isinstance(p, (str, Path)):
        df = pd.read_parquet(p, memory_map=True)
        return _extract_features_from_df(df, feature_columns)
    else:
        embeddings = []
        for p_ in p:
            df = pd.read_parquet(p_, memory_map=True)
            embeddings.append(_extract_features_from_df(df, feature_columns))
        return torch.cat(embeddings)


def _extract_features_from_df(
    df: pd.DataFrame, feature_columns: Optional[List[str]] = None
) -> Tensor:
    """Extract features from DataFrame into a tensor

    Args:
        df: DataFrame containing feature columns
        feature_columns: List of feature column names. If None, columns matching 'feature_*' will be used

    Returns:
        Tensor of embeddings
    """
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col.startswith("feature_")]

    if not feature_columns:
        raise ValueError("No feature columns found in DataFrame")

    # Sort columns to ensure consistent ordering
    feature_columns = sorted(
        feature_columns, key=lambda x: int(x.removeprefix("feature_"))
    )

    # Extract feature columns and convert to tensor
    features = df[feature_columns].values
    return torch.tensor(features)


class EmbeddingNotFoundError(Exception):
    pass


# In case other Embedding Loader classes must be implemented
class EmbeddingLoader(Protocol):
    def load(self, __identifier: Union[str, Iterable[str]]) -> Tensor:
        ...

    def lookup_embeddings_filepath(self, embedding_filename: str) -> Optional[Path]:
        ...


class FileSystemEmbeddingLoader(EmbeddingLoader):
    """Loads embeddings files."""

    def __init__(
        self,
        embeddings_dir: Union[str, Path],
        load_func: Callable[[Union[Path, Iterable[Path]]], Tensor] = load_torch,
        extension: str = '.pt',
    ):
        """Initialize embedding loader.

        Args:
            embeddings_dir: Directory expected to contain all embedding files.
            load_func: Reads one or more embedding files, concatenates them in the latter case.
            extension: Embeddings file extension. Defaults to `.pt`.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.load_func = load_func
        self.extension = extension

    def load(self, embedding_filename_or_names: Union[str, Iterable[str]]) -> Tensor:
        """
        Loads embeddings for a given group name.

        Args:
            embedding_filename_or_names: Identifies the name(s) of the embeddings filepaths to be loaded.
        """
        if isinstance(embedding_filename_or_names, str):
            embeddings = self.load_func(
                self.lookup_embeddings_filepath(embedding_filename_or_names)
            )
        else:
            embeddings = self.load_func(
                self.lookup_embeddings_filepath(p) for p in embedding_filename_or_names
            )
        return embeddings

    def lookup_embeddings_filepath(self, embedding_filename: str) -> Path:
        """
        Finds the embedding filepath.

        Args:
            embedding_filename: The name of the embeddings file

        Raises:
            EmbeddingNotFoundError: If no embeddings were found.

        Returns:
            The path to the embeddings file.
        """
        embedding_path = self.embeddings_dir / (embedding_filename + self.extension)
        if not embedding_path.exists():
            raise EmbeddingNotFoundError(f'embedding_path {embedding_path} does not exist')

        return embedding_path


class ParquetEmbeddingLoader(EmbeddingLoader):
    """Loads embeddings from parquet files containing feature columns."""

    def __init__(
        self,
        embeddings_dir: Union[str, Path],
        feature_columns: Optional[List[str]] = None,
        extension: str = ".parquet",
    ):
        """Initialize parquet embedding loader.

        Args:
            embeddings_dir: Directory expected to contain all parquet embedding files.
            feature_columns: List of feature column names. If None, columns matching 'feature_*' will be used.
            extension: Embeddings file extension. Defaults to `.parquet`.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.feature_columns = feature_columns
        self.extension = extension

    def load(self, embedding_filename_or_names: Union[str, Iterable[str]]) -> Tensor:
        """
        Loads embeddings for a given group name from parquet files.

        Args:
            embedding_filename_or_names: Identifies the name(s) of the embeddings filepaths to be loaded.
        """
        if isinstance(embedding_filename_or_names, str):
            path = self.lookup_embeddings_filepath(embedding_filename_or_names)
            embeddings = load_parquet(path, self.feature_columns)
        else:
            paths = [
                self.lookup_embeddings_filepath(p) for p in embedding_filename_or_names
            ]
            embeddings = load_parquet(paths, self.feature_columns)
        return embeddings

    def lookup_embeddings_filepath(self, embedding_filename: str) -> Path:
        """
        Finds the embedding filepath.

        Args:
            embedding_filename: The name of the embeddings file

        Raises:
            EmbeddingNotFoundError: If no embeddings were found.

        Returns:
            The path to the embeddings file.
        """
        embedding_path = self.embeddings_dir / (embedding_filename + self.extension)
        if not embedding_path.exists():
            raise EmbeddingNotFoundError(
                f"embedding_path {embedding_path} does not exist"
            )

        return embedding_path
