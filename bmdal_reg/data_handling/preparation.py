import copy
import logging
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .datasets import GKDatasetDF
from .datautils import target_keys_store, train_keys_store

def get_percentiles(
    dataset: GKDatasetDF,
    fluxes: List[str],
    percentiles: List[float],
    scaler: StandardScaler,
) -> Dict[str, float]:
    """Function that generates the percentiles of the fluxes of interest

    Args:
        dataset (GKDatasetDF): The dataset that contains the fluxes
        fluxes (list): The fluxes of interest
        percentiles (list): A list of floats from 0 to 100
        scaler (StandardScaler): The scaler the data has been scaled with

    Returns:
        dict: flux-percentiles pairs
    """

    ds = copy.deepcopy(dataset)
    ds.scale(scaler, unscale=True)
    out: dict = {f: {} for f in fluxes}
    for flux in fluxes:
        numpyperc = np.percentile(ds.data.loc[:, flux], percentiles)
        out[flux] = numpyperc
    return out


def load_data(    
    train_path: str,
    valid_path: str,
    test_path: str,
    pool_path: str
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the data from the given paths.

    Args:
        train_path (str): The path where the training data file is
        valid_path (str): The path where the validation data file is
        test_path (str): The path where the testing data file is
        pool_path (str): The path where the pool data file is
    """

    try:
        logging.debug(f"loading data from {train_path}")
        if isinstance(train_path, dict):
            train_data = {}
            train_data["Classifier"] = pd.read_pickle(train_path["Classifier"])
            train_data["Regressor"] = pd.read_pickle(train_path["Regressor"])
        else:
            train_data = pd.read_pickle(train_path)
        validation_data = pd.read_pickle(valid_path)
        test_data = pd.read_pickle(test_path)
        pool_data = pd.read_pickle(pool_path)
    except Exception:
        import pickle5 as pkl

        with open(train_path, "rb") as f:
            train_data = pkl.load(f)
        with open(valid_path, "rb") as f:
            validation_data = pkl.load(f)
        with open(test_path, "rb") as f:
            test_data = pkl.load(f)
        with open(pool_path, "rb") as f:
            pool_data = pkl.load(f)


    logging.debug(f'columns {train_data.columns}')
    return train_data, validation_data, test_data, pool_data


def clean_and_tidyup(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    pool_data: pd.DataFrame,        
    fluxes: List,
    gkmodel: str,
    labels_unavailable: str,
    **cfg
):
    """Removes nans, keeps only relevant columns from data and casts to a GKDatasetDF object

    Args:
        train_data (pd.DataFrame): The training data
        validation_data (pd.DataFrame): Validation data
        test_data (pd.DataFrame): Test data
        pool_data (pd.DataFrame): Pool data
        fluxes (List): The fluxes
        gkmodel (str): The model used
        labels_unavailable (str): Availability of labels
    """    
    leading_flux = fluxes[0]
    input_keys = train_keys_store(gkmodel)
    fluxes_temp = target_keys_store(leading_flux)
    keep_keys = input_keys + fluxes_temp

    train_data = train_data[keep_keys]
    validation_data = validation_data[keep_keys]
    test_data = test_data[keep_keys]

    if not labels_unavailable:
        pool_data = pool_data[keep_keys]
    else:
        pool_data = pool_data[input_keys]
        for flux in fluxes:
            pool_data[
                flux
            ] = 1  # dummy variable, must be >0, if nan the dataset will be emptied after the call to all_clean below

    train_dataset = GKDatasetDF(train_data, leading_flux=leading_flux, gkmodel=gkmodel)
    valid_dataset = GKDatasetDF(validation_data, leading_flux=leading_flux, gkmodel=gkmodel)
    test_dataset = GKDatasetDF(test_data, leading_flux=leading_flux, gkmodel=gkmodel)
    pool_dataset = GKDatasetDF(pool_data, leading_flux=leading_flux, gkmodel=gkmodel, is_pool=True)

    train_dataset.all_clean(define_stability=True)
    valid_dataset.all_clean(define_stability=True)
    test_dataset.all_clean(define_stability=True)
    pool_dataset.all_clean(define_stability=True)
    

    logging.debug(f"After cleaning: are there nans in the train? {train_dataset.data.query('stable==1').isnull().values.any()}")
    logging.debug(f"After cleaning: are there nans in the candidates? {pool_dataset.data.query('stable==1').isnull().values.any()}")
    logging.debug(f"After cleaning: are there nans in the valid? {valid_dataset.data.query('stable==1').isnull().values.any()}")
    logging.debug(f"After cleaning: are there nans in the test? {test_dataset.data.query('stable==1').isnull().values.any()}")

    if labels_unavailable:
        for flux in fluxes:
            pool_dataset.data.loc[:, flux] = np.nan

    keep_keys_final = input_keys + fluxes + [train_dataset.stability_name]
    train_dataset.data = train_dataset.data[keep_keys_final]
    valid_dataset.data = valid_dataset.data[keep_keys_final]
    test_dataset.data = test_dataset.data[keep_keys_final]
    pool_dataset.data = pool_dataset.data[keep_keys_final]

    return train_dataset, valid_dataset, test_dataset, pool_dataset


def scale_data(
    train: GKDatasetDF,
    valid: GKDatasetDF,
    test: GKDatasetDF,
    pool_data: GKDatasetDF,
    scaler: StandardScaler = None,
    unscale: bool = False,
) -> Tuple[
    GKDatasetDF,
    GKDatasetDF,
    GKDatasetDF,
    GKDatasetDF,
    StandardScaler,
]:
    """Scales the dataset with a given scaler. If the scaler is not given, it gets fit to the regression data.

    Args:
        train (GKDatasetDF): The training data
        valid (GKDatasetDF): The validation data
        test (GKDatasetDF): The test data
        pool_data (GKDatasetDF): The unlabelled pool
        scaler (StandardScaler, optional): The scaler with which to scale the data. Defaults to None.
        unscale (bool, optional): Whether to scale or unscale the datasets. Defaults to False.

    Returns:
        Tuple[GKDatasetDF,GKDatasetDF,GKDatasetDF,GKDatasetDF, StandardScaler]: The scaled data in the same order, and the scaler
    """
    if (
        scaler is None
    ):  # --- ensures that future tasks can be scaled according to scaler of previous tasks
        scaler = StandardScaler()
        scaler.fit(train.data.drop(columns=train.stability_name, axis=1))
    train.scale(scaler, unscale=unscale)
    valid.scale(scaler, unscale=unscale)
    test.scale(scaler, unscale=unscale)
    pool_data.scale(scaler, unscale=unscale)
    return train, valid, test, pool_data, scaler


def get_data(
    cfg: Dict[Any, Any],
    scaler: StandardScaler = None,
) -> Tuple[
    GKDatasetDF,
    GKDatasetDF,
    GKDatasetDF,
    GKDatasetDF,
    StandardScaler,
]:
    """Retrieves the datasets, cleans them and casts them in the GKDatasetDF format.

    Args:
        cfg (dict): The config dictionary
        scaler (StandardScaler, optional): The scaler to be used. Defaults to None.

    Returns:
        Tuple[GKDatasetDF,GKDatasetDF,GKDatasetDF,GKDatasetDF,GKDatasetDF,GKDatasetDF,GKDatasetDF, StandardScaler]: The datasets ready to use and the scaler with which the data has been scaled.
    """

    PATHS = cfg["data"]
    fluxes = cfg['fluxes']
    logging.info(f"Reading data from: {PATHS}")
    

    train, valid, test, pool_data = load_data(
        train_path=PATHS["train"],
        valid_path=PATHS["validation"],
        test_path=PATHS["test"],
        pool_path=PATHS["pool"]
    )

    leading_flux = fluxes[0]

    if not cfg['use_classifier']:
        if not cfg['use_all_outputs']:
            train_data = train_data.query(f"{leading_flux}>0")
            validation_data = validation_data.query(f"{leading_flux}>0")
            test_data = test_data.query(f"{leading_flux}>0")
        else:
            if len(fluxes) > 1:
                raise ValueError(
                    "use_all_outputs to be used only with the leading flux"
                )

    if cfg['denormalise']:
        for flux in fluxes:
            if flux != leading_flux:
                train_data.loc[:, flux] = (
                    train_data.loc[:, flux].values
                    * train_data.loc[:, leading_flux].values
                )
                pool_data.loc[:, flux] = (
                    pool_data.loc[:, flux].values
                    * pool_data.loc[:, leading_flux].values
                )
                validation_data.loc[:, flux] = (
                    validation_data.loc[:, flux].values
                    * validation_data.loc[:, leading_flux].values
                )
                test_data.loc[:, flux] = (
                    test_data.loc[:, flux].values
                    * test_data.loc[:, leading_flux].values
                )

    train, valid, test, pool_data = clean_and_tidyup(
        train, valid, test, pool_data, **cfg
    )
    
    logging.info("Finishing up...")
    if len(train) > cfg["train_size"]:
        train = train.sample(cfg["train_size"])
    # --- valid sets
    if len(valid) > cfg["valid_size"]:
        valid = valid.sample(cfg["valid_size"])

    # --- test sets
    if len(test) > cfg["test_size"]:
        test = test.sample(cfg["test_size"])

    train, valid, test, pool_data, scaler = scale_data(
        train, valid, test, pool_data, scaler=scaler
    )


    return train, valid, test, pool_data, scaler
