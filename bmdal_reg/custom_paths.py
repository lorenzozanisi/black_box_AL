# This file allows to configure where to save data, results, plots etc.
class CustomPaths:
    # path where downloaded data sets will be saved
    data_path = '/rds/project/iris_vol2/rds-ukaea-ap001/ir-zani1/qualikiz/UKAEAGroupProject/data'
    # path where benchmark results will be saved
    results_path = '/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/black_box_AL/results'
    # path where plots and tables will be saved
    plots_path = '/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/black_box_AL/data/plots'
    # path where benchmark results can be cached in a more efficient format such that they load faster
    cache_path = '/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/black_box_AL/data/cache'


def get_data_path():
    return CustomPaths.data_path


def get_results_path():
    return CustomPaths.results_path


def get_plots_path():
    return CustomPaths.plots_path


def get_cache_path():
    return CustomPaths.cache_path

