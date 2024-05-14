import numpy as np
import pandas as pd
from adept.data.datasets.datasets import GKDatasetDF
import logging

def init_streaming_datasets(leading_flux: str, gkmodel: str, **cfg):
    """
    Returns three empty GKDatasetDF objects
    """
    train_master = GKDatasetDF(pd.DataFrame({}), leading_flux=leading_flux, gkmodel=gkmodel)
    valid_master = GKDatasetDF(pd.DataFrame({}), leading_flux=leading_flux, gkmodel=gkmodel)
    test_master = GKDatasetDF(pd.DataFrame({}), leading_flux=leading_flux, gkmodel=gkmodel)
    return train_master, valid_master, test_master



def load_streaming_mock(streaming_mockup_datafile: str, num_campaigns: int = 10, **cfg):
    """Loads the mockedup streaming data. Mockup is done on the y axis of a 2D PCA representation as shown in the paper.

    Args:
        streaming_mockup_datafile (str): The file to load the data from. Expects csv format.
        num_campaigns (int, optional): The number of mocked-up campaigns based on the PCA reduction of the dataset. Defaults to 10.
    """
    data_master = pd.read_csv(streaming_mockup_datafile)
    data_master = data_master.query("cluster=='left' & is_hmode==False")
    nonzeros = data_master.query('machtor!=0').index
    data_master = data_master.drop(index=nonzeros)
    logging.debug(f'data master: {data_master}')    
    PCA_ybins = np.linspace(-2., 2., num_campaigns + 1)[::-1] # --- reversed because low bin numbers equate high plasma performance
    for i in range(len(PCA_ybins) - 1):
        idx = data_master.query(f"pca_y<{PCA_ybins[i]} & pca_y>{PCA_ybins[i+1]}").index
        data_master.loc[idx, "num_campaign"] = i
    logging.debug(f'after cleaning data master: {data_master}')
    
    return data_master


def sample_from_campaign(campaign_data: pd.DataFrame):
    """Selects a random shot in the campaign number requested.

    Args:
        campaign_data (pd.DataFrame): The dataset from which to extract the campaign.
    """
    num_shots = campaign_data["shot_num"].unique()
    if len(num_shots)>0:
        sampled_shot = np.random.choice(num_shots)
        data = campaign_data.query(f"shot_num=={sampled_shot}")
    else:
        return -1
    return data
