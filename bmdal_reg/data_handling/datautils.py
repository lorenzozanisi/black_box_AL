from typing import List, Dict

import pandas as pd

output_dict: Dict[str, List[float]] = {
    "train_loss": [],
    "val_loss": [],
    "val_loss_unscaled": [],
    "test_loss_unscaled": [],
    "popback": [],
    "popback_nozeros": [],
    "R2_nozeros": [],
    "test_loss_unscaled_nozeros": [],
    "num_train_points": [],
    "n_iterations": [],
    "loss_0_5": [],
    "loss_30_35": [],
    "loss_50_55": [],
    "loss_80_85": [],
    "loss_95_100": [],
    "class_train_loss": [],
    "class_val_loss": [],
    "class_train_acc": [],
    "class_val_acc": [],
    "class_missed_acc": [],
    "accuracy": [],
    "test_class_loss": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "auc": [],
    "R2": [],
}

jet_keys: List[str] = ["wall_material_index", "is_hmode", "discharge_phase_index"]

leading_fluxes: List[str] = ["leading", "efeetg_gb", "efiitg_gb", "efetem_gb"]

particle_fluxes: List[str] = [
    "pfiitg_gb_div_efiitg_gb",
    "pfeitg_gb_div_efiitg_gb",
    "pfetem_gb_div_efetem_gb",
    "pfitem_gb_div_efetem_gb",
]
#'pfietg_gb_div_efiitg_gb']#, 'pfeetg_gb_div_efeetg_gb']

momentum_fluxes: List[str] = [
    "vfiitg_gb_div_efiitg_gb",
    "vfitem_gb_div_efetem_gb",
]  # ,'vfeitg_gb_div_efiitg_gb',
#'vfetem_gb_div_efetem_gb', 'vfitem_gb_div_efetem_gb',
#'vfietg_gb_div_efiitg_gb']#, 'vfeetg_gb_div_efeetg_gb']

heat_fluxes: List[str] = [
    "efeitg_gb_div_efiitg_gb",
    "efitem_gb_div_efetem_gb",
    "efietg_gb_div_efeetg_gb",
]

target_keys = leading_fluxes + particle_fluxes + momentum_fluxes + heat_fluxes


def train_keys_store(gkmodel: str, return_all: bool = False) -> List[str]:
    """
    A function that given a gyrokinetic model setup returns the input keys for that particular model.

    Args:
        gkmodel (str). Currently supports 'QLK15D' and 'TGLF'.
    Returns:
        List: list with the inputs for that specific gkmodel

    """
    if gkmodel == "QLK15D":
        return [
            "ane",
            "ate",
            "autor",
            "machtor",
            "x",
            "zeff",
            "gammae",
            "q",
            "smag",
            "alpha",
            "ani1",
            "ati0",
            "normni1",
            "ti_te0",
            "lognustar",
        ]
    if gkmodel == "QLKStreaming":
        return [
            "ane",
            "ate",
            "x",
            "zeff",
            "q",
            "smag",
            "alpha",
            "ani1",
            "ati0",
            "normni1",
            "ti_te0",
            "lognustar",
        ]
    if gkmodel == "TGLF":
        if return_all:
            return [
                "RLNS_1",
                "RLNS_2",
                "RLTS_1",
                "RLTS_2",
                "TAUS_2",
                "RMIN_LOC",
                "DRMAJDX_LOC",
                "Q_LOC",
                "DQDR",
                "BETAE",
                "XNUE",
                "ZEFF",
                "ZS_1",
                "TAUS_1",
                "AS_1",
                "VPAR_1",
                "VPAR_SHEAR_1",
                "ZS_2",
                "MASS_2",
                "AS_2",
                "VPAR_2",
                "VPAR_SHEAR_2",
                "RMAJ_LOC",
                "ZMAJ_LOC",
                "KX0_LOC",
                "KAPPA_LOC",
                "S_KAPPA_LOC",
                "DELTA_LOC",
                "ZETA_LOC",
                "S_ZETA_LOC",
                "P_PRIME_LOC",
            ]
        else:
            return [
                "RLNS_1",
                "RLTS_1",
                "RLTS_2",
                "TAUS_2",
                "RMIN_LOC",
                "DRMAJDX_LOC",
                "Q_LOC",
                "DQDR",
            ]
    elif gkmodel == "TGLF_ES":
        return [
            "RLNS_1",
            "RLTS_1",
            "RLTS_2",
            "TAUS_2",
            "RMIN_LOC",
            "DRMAJDX_LOC",
            "Q_LOC",
            "SHAT",
            "XNUE",
            "KAPPA_LOC",
            "S_KAPPA_LOC",
            "DELTA_LOC",
            "S_DELTA_LOC",
        ]
    elif gkmodel == "TGLF_EM":
        return [
            "RLNS_1",
            "RLTS_1",
            "RLTS_2",
            "TAUS_2",
            "RMIN_LOC",
            "DRMAJDX_LOC",
            "Q_LOC",
            "SHAT",
            "XNUE",
            "KAPPA_LOC",
            "S_KAPPA_LOC",
            "DELTA_LOC",
            "S_DELTA_LOC",
            "BETAE",
        ]


def target_keys_store(leading_flux: str = "efiitg_gb") -> List[str]:
    """Stores the target keys for a given turbulence mode

    Args:
        leading flux (str, optional): The leading flux of the given turbulence mode. Can be "efiitg_gb","efetem_gb","efeetg_gb". Defaults to 'efiitg_gb'.

    Returns:
        Targets (list): All the targets relevant to that leading flux.
    """
    if leading_flux == "efiitg_gb":
        targets = [leading_flux] + [
            t for t in target_keys if t.find("itg") != -1 and t.find("div") != -1
        ]
    elif leading_flux == "efetem_gb":
        targets = [leading_flux] + [
            t for t in target_keys if t.find("tem") != -1 and t.find("div") != -1
        ]
    elif leading_flux == "efeetg_gb":
        targets = [leading_flux] + [
            t for t in target_keys if t.find("etg") != -1 and t.find("div") != -1
        ]
    elif leading_flux == "leading":
        targets = TGLF_target_keys
    else:
        raise ValueError(f"Leading flux: {leading_flux} not recognised")

    return targets


TGLF_target_keys = ["leading", "efe_gb", "efi_gb", "pfi_gb"]


def remove_all_nans(
    ds: pd.DataFrame, keys: List[str], leading_flux: str
) -> pd.DataFrame:
    """Removes the NaNs from a dataset. However, the NaNs in columns where the leading flux is non-zero should not be removed, see https://arxiv.org/pdf/1911.05617.pdf

    Args:
        ds (pd.DataFrame): The dataset to be processed
        keys (list): A list of the columns to process
        leading_flux (str): The leading flux of the turbulence mode considered, see Appendix D of https://arxiv.org/pdf/1911.05617.pdf

    Returns:
        pd.DataFrame: The processed dataset
    """
    ds = ds.query(f"{leading_flux}>=0")
    for col in keys:
        if col != leading_flux:
            fluxes_nans = ds[col].isna() & ds[leading_flux] > 0
            ds = ds[~fluxes_nans]

    return ds
