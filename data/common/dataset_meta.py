from dataclasses import dataclass

@dataclass
class DatasetMeta:
    data_path: str
    feature_description: dict
    input_size: list