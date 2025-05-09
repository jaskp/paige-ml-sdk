import logging
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Optional, Set, Tuple, Union, cast

import pandas as pd
from torch.utils.data import Dataset

from paige.ml_sdk.dataset_universe.datasets.fit import EmbeddingAggregatorFitDatasetItem
from paige.ml_sdk.dataset_universe.embedding_loader import (
    FileSystemEmbeddingLoader,
    H5EmbeddingLoader,
    ParquetEmbeddingLoader,
    ZarrEmbeddingLoader,
)

logger = logging.getLogger(__name__)


PathLike = "Union[str, Path[str]]"


class EmbeddingDataset(Dataset[EmbeddingAggregatorFitDatasetItem]):
    def __init__(
        self,
        dataset: pd.DataFrame,
        embeddings_dir: PathLike,
        label_columns: Set[str],
        embeddings_filename_column: str,
        label_missing_value: int,
        group_column: Optional[str] = None,
        validate_all_embeddings_exist: bool = True,
        filename_extension: str = '.pt',
    ) -> None:
        """
        Dataset to iterate over rows in a dataframe and return embedding Tensor from an
        EmbeddingLoader.

        Args:
            dataset: A dataframe with each row containing a WSI file path and its labels.
            embeddings_dir: Name of the dirtectory containing the embeddings.
            label_columns: The names of the columns to select as labels.
            embeddings_filename_column: The name of the column which specifies, for a
                given row (i.e. a given group), the path to its embedding file(s).
            label_missing_value: The value to replace NaN values with in the instance mask
                map.
            group_column: The name of grouping column (not the index).
            validate_all_embeddings_exist: Whether or not to validate existence of embedding
                files during initialization.
            filename_extension: Embeddings filename extension. Defaults to `.pt`
        """
        self._df = dataset  # use the `df` property instead, which adds an index column
        self.label_columns = label_columns
        self.label_missing_value = label_missing_value
        self.embeddings_filename_column = embeddings_filename_column
        self.group_column = group_column or embeddings_filename_column
        match filename_extension:
            case ".parquet":
                self.embedding_loader = ParquetEmbeddingLoader(
                    embeddings_dir=embeddings_dir
                )
            case ".h5":
                self.embedding_loader = H5EmbeddingLoader(embeddings_dir=embeddings_dir)
            case ".zarr":
                self.embedding_loader = ZarrEmbeddingLoader(
                    embeddings_dir=embeddings_dir
                )
            case _:
                self.embedding_loader = FileSystemEmbeddingLoader(
                    embeddings_dir=embeddings_dir, extension=filename_extension
                )

        # useful to disable when we need to power through.
        if validate_all_embeddings_exist:
            self._validate_all_embeddings_exist()

    @classmethod
    def from_filepath(
        cls,
        dataset: PathLike,
        embeddings_dir: PathLike,
        label_columns: Set[str],
        embeddings_filename_column: str,
        label_missing_value: int,
        group_column: Optional[str] = None,
        validate_all_embeddings_exist: bool = True,
        filename_extension: str = ".pt",
        mode: Literal["csv", "parquet", "xlsx"] = "csv",
    ) -> "EmbeddingDataset":
        """Instantiates an EmbeddingDataset from a dataset filepath"""
        reader = (
            pd.read_csv
            if mode == "csv"
            else pd.read_parquet
            if mode == "parquet"
            else pd.read_excel
        )
        return cls(
            reader(dataset),
            embeddings_dir,
            label_columns,
            embeddings_filename_column,
            label_missing_value,
            group_column,
            validate_all_embeddings_exist,
            filename_extension,
        )

    @cached_property
    def index(self) -> str:
        """Returns The name of the column_name index."""
        return f'{self.group_column}_index'

    @cached_property
    def df(self) -> pd.DataFrame:
        """
        This cached property returns a copy of the original dataframe with an index column added.
        This index column is used in __getitem__ to select rows belonging to the appropriate group.

        Returns:
            The dataframe with the index column attached.
        """
        df = self._df.copy(deep=True)
        self._columns_sanity_check(
            df, self.label_columns, self.embeddings_filename_column, self.group_column
        )
        # This creates a column like `[0,1,2,...]`
        df[self.index] = df[self.group_column].astype('category').cat.codes.astype('int64')
        return df

    def __getitem__(self, group_index: int) -> EmbeddingAggregatorFitDatasetItem:
        """
        Return the embedding and its labels for a given group_index.

        Args:
            group_index : The index of a group to return the embedding for.

        Returns:
            An item with embedding and labels.
        """
        labels, embedding_file_names = self._get_from_df(group_index)

        # Load the embedding for this group.
        embeddings = self.embedding_loader.load(embedding_file_names)
        label_map = dict(zip(self.label_columns, labels))
        instance_mask_map = {k: v != self.label_missing_value for k, v in label_map.items()}

        return EmbeddingAggregatorFitDatasetItem(
            group_index=group_index,
            embeddings=embeddings,
            label_map=label_map,
            instance_mask_map=instance_mask_map,
        )

    def _validate_all_embeddings_exist(self) -> None:
        """
        Validate that all wsi_file_path in the dataframe have embeddings associated to them.
        This allows us to check before starting the training loop if all embeddings we expect
        have been generated.

        Raises:
            EmbeddingNotFoundError: If some embeddings are missing.
        """
        for wsi_file_str in self.df[self.embeddings_filename_column]:
            self.embedding_loader.lookup_embeddings_filepath(wsi_file_str)

    @staticmethod
    def _columns_sanity_check(
        df: pd.DataFrame,
        label_columns: Set[str],
        embeddings_filename_column: str,
        group_column: str,
    ) -> None:
        for column in (*label_columns, embeddings_filename_column, group_column):
            if column not in df.columns:
                raise AssertionError(
                    f'column `{column}` not present in table with columns {df.columns.tolist()}'
                )

        if df[embeddings_filename_column].nunique() != len(df):
            raise AssertionError('embedding filenames should be unique.')

    def _get_from_df(
        self, group_index: int
    ) -> Tuple[Tuple[Union[int, float, bool], ...], List[str]]:
        wsi_in_group = self.df[self.df[self.index] == group_index]
        labels = cast(
            Tuple[Union[int, float, bool], ...],
            tuple(wsi_in_group[[*self.label_columns]].iloc[0]),
        )
        embedding_file_names = wsi_in_group[self.embeddings_filename_column].unique()
        return labels, embedding_file_names.tolist()

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return self.df[self.index].nunique()
