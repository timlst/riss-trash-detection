import os

from detectron2.config import get_cfg
from detectron2.model_zoo import get_config, get_config_file


def get_config_from_path(filepath):
    """
    Loads a CfgNode from filepath: Can either refer to system or model zoo, will raise Exception if neither is available.
    :param filepath: relative filepath, either for file system or model zoo
    :return: CfgNode
    """
    if os.path.exists(filepath):
        cfg = get_cfg()
        cfg.merge_from_file(filepath)
    else:
        cfg = get_config(get_config_file(filepath))
    return cfg