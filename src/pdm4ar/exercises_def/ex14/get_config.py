from typing import List


def get_config() -> List[str]:
    "Choose manually which .yaml file to load and test."

    config_files = [
        "config_1.yaml",  # pass
        # "config_2.yaml",  # pass
        "config_3.yaml",  # pass
        # "c_generic_1.yaml",  # pass
        # "c_generic_2.yaml",  # Not enough time -> no crash, not all goals
        # "c_generic_4.yaml",  # pass
        "c_generic_6.yaml",  # pass
        # "c_generic_7.yaml",  # pass
        # "c_generic_8.yaml",  # pass
        # "c_generic_9.yaml",  # pass
        # "c_generic_10.yaml",  # pass, not enough time
        # "c_generic_11.yaml",  # pass
        # "c_crowded_center.yaml",  # crash -> investigate possible misplanning
        # "c_narrow_passages.yaml",  # pass
    ]

    return config_files
