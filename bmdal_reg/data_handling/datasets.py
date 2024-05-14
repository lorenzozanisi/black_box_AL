from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .datautils import (
    leading_fluxes,
    remove_all_nans,
    train_keys_store,
)


class GKDataset(Dataset):  # type: ignore
    """Dataset class that handles classification and regression tasks at the same time

    Args:
        X (np.array): The input data
        y  (np.array): The labels of a classification task
        z (np.array): The labels of a regression task
    """

    def __init__(self, X, y, z=None, indices=None) -> None:  # type: ignore
        self.X = X
        self.y = y
        self.z = z
        self.indices = indices
        return

    # number of rows in the dataset
    def __len__(self):  # type: ignore
        return len(self.y)

    # get a row at an index
    def __getitem__(self, idx):  # type: ignore
        return self.X[idx], self.y[idx], self.z[idx], self.indices[idx]


class GKDatasetDF(Dataset):  # type: ignore
    """Dataset class that handles classification and regression tasks at the same time.

    It handles Pandas DataFrames and it is used as the main object with which the Active Learning pipeline interacts.

    Args:
        df (pd.DataFrame): The DataFrame object that contains the input-output data pairs
        leading_flux (str): The leading flux of the turbulence mode considered, see Appendix D of https://arxiv.org/abs/1911.05617
        gkmodel (str): The GK model being used. Default is "QLK15D". The input variables will be different for different GK models. Defaults to "QLK15D".
        is_pool (bool): Whether the dataset is from the unlabelled pool. If true, the labels are hidden (if available)

    """

    def __init__(
        self, df: pd.DataFrame, leading_flux: str, gkmodel: str, is_pool: str = False
    ) -> None:
        self.data = df
        self.idx = df.index
        self.leading_flux = leading_flux
        self.stability_name = "stable"
        self.train_keys = train_keys_store(gkmodel)
        self.gkmodel = gkmodel
        self.is_pool = is_pool
        return

    def all_clean(self, define_stability: bool = False) -> None:
        """Cleans the dataset from NaNs and gives the option to add the stability label to the data, which is needed for the classifier.

        Args:
            define_stability (bool, optional): Gives the option to add the stability label to the data, which is needed for the classifier. Defaults to False.
        """
        dataset = self.data
        dataset = dataset.dropna(subset=[self.leading_flux])
        dataset = remove_all_nans(
            dataset, keys=self.data.columns, leading_flux=self.leading_flux
        )
        if define_stability:
            dataset[self.stability_name] = np.where(
                dataset[self.leading_flux] != 0, 1, 0
            )

        self.data = dataset
        return

    def scale(self, scaler: StandardScaler, unscale: bool = False) -> None:
        """Scales the data according to a scaler provided. The scaler must be fit before passing it here. The data must have been processed by `all_clean` with `define_stability=True`.

        Args:
            scaler (StandardScaler): The scaler fit to the training data
            unscale (bool, optional): Whether to scale (False) or to unscale (True). Defaults to False.
        """
        index = self.data.index
        keep_col = self.data[self.stability_name].values
        columns = [col for col in self.data.columns if col != self.stability_name]

        if not unscale:
            self.data = scaler.transform(
                self.data.drop(columns=self.stability_name, axis=1)
            )
        else:
            self.data = scaler.inverse_transform(
                self.data.drop(columns=self.stability_name, axis=1)
            )

        self.data = pd.DataFrame(self.data, columns=columns, index=index)
        self.data[self.stability_name] = keep_col

        return

    def sample(self, sample_size: int):  # type: ignore
        """Returns another GKDatasetDF object initialised with a subsample of size sample_size taken at random.

        Args:
            sample_size (int): The sample size of resampling

        Returns:
            GKDatasetDF: The GKDatasetDF object initialised with a subsample of size sample_size taken at random.
        """
        return GKDatasetDF(
            self.data.sample(sample_size),
            leading_flux=self.leading_flux,
            gkmodel=self.gkmodel,
            is_pool=self.is_pool,
        )

    def add(self, dataset: pd.DataFrame) -> None:
        """Concatenates the current data with new data.

        Args:
            dataset (pd.DataFrame): The data to be concatenated to the current data
        """
        self.data = pd.concat([self.data, dataset.data], axis=0)
        return

    def set_output(self, flux: str, test: bool = False):
        """Sets which flux the surrogate will be fit to.

        Args:
            flux (str): The flux to be fit to
            test (bool): whether the data is being set up for testing, in which case only unstable data needs to be used for the regressor of the leading flux
        """
        if ((not self.is_pool) and (flux not in leading_fluxes)) or test:
            # --- makes sure that the NaNs are filtered out in the flux ratios
            # --- this option needs to be bypassed for candidate evaluation as the true stability label is unkonwn
            # --- the self.data pool needs to be considered in its entirety, so this if clause must be ignored
            data_now = self.data.query(f"{self.stability_name}==1")
        else:
            data_now = self.data

        self.x = torch.Tensor(data_now[self.train_keys].values)
        self.y = torch.Tensor(data_now[flux].values)
        self.z = torch.Tensor(data_now[self.stability_name].values)
        self.idx = torch.tensor(data_now.index.values, dtype=torch.int32)
        return

    def remove(self, indices: Any) -> None:
        """Removes data corresponding to the given indices

        Args:
            indices (pd.Index): indices to be dropped
        """
        indices = [idx for idx in indices if idx in self.data.index]
        self.data.drop(index=indices, inplace=True)
        return

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, ind):  # type : ignore
        return self.x[ind], self.y[ind], self.z[ind], self.idx[ind]


class StreamingGKDatasetDF(GKDatasetDF):
    """A dataset for handling mocked streaming data from the JETExp-mockstream dataset.

    Args:
        df (pd.DataFrame): The DataFrame object that contains the input-output data pairs
        leading_flux (str): The leading flux of the turbulence mode considered, see Appendix D of https://arxiv.org/abs/1911.05617
        gkmodel (str): The GK model being used. Default is "QLK15D". The input variables will be different for different GK models. Defaults to "QLK15D"

        num_campaigns (int, optional): The number of mocked-up campaigns based on the PCA reduction of the dataset. Defaults to 10.
        num_shots_per_campaign (int, optional): How many shots are sampled for a given campaign. Defaults to 1.
    """

    def __init_(
        self,
        df: pd.DataFrame,
        leading_flux: str,
        gkmodel: str = "QLK15D",
        num_campaigns: int = 10,
        num_shots_per_campaign: int = 1,
    ):
        super().__init__(self, df=df, leading_flux=leading_flux, gkmodel=gkmodel)
        # --- Renaming self.data for convenience
        self.data_master = self.data
        del self.data
        # --- Creates index for each bin in the y pca component, a proxy for an experimental campaign
        PCA_ybins = np.linspace(-3, 3, num_campaigns + 1)
        for i in range(len(PCA_ybins) - 1):
            idx = self.data_master.query(
                f"PCA_y>{PCA_ybins[i]} & PCA_y<{PCA_ybins[i+1]}"
            ).index
            self.data_master.loc[idx, "num_campaigns"] = i

    def sample_campaign(self, campaign_num: int):
        """Selects a random shot in the campaign number requested.

        Args:
            campaign_num (int): The campaign number requested.
        """
        campaign_data = self.data_master.query(f"num_campaign={campaign_num}")
        num_shots = campaign_data.unique()["shot_num"]
        sampled_shot = np.random.choice(num_shots)
        self.data = self.campaign_data.query(f"shot_num=={sampled_shot}")
