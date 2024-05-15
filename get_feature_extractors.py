from finrl_myself.feature_extractor.Base import BaseFeaturesExtractor
from finrl_myself.feature_extractor.Flatten import FlattenExtractor
from typing import Dict,Type


def get_feature_extractor(feature_extractor_aliase: str) -> Type[BaseFeaturesExtractor]:

    feature_extractors_aliases: Dict[str, Type[BaseFeaturesExtractor]] = {"flatten": FlattenExtractor,}

    if feature_extractor_aliase in feature_extractors_aliases:
        feature_extractor = feature_extractors_aliases[feature_extractor_aliase]
        if feature_extractor_aliase=='flatten':
            print(f'feature_extractor: FlattenExtractor !')

    else:
        raise ValueError(f"Feature Extractor {feature_extractor_aliase} unknown")



    return feature_extractor

# get_feature_extractor('flatten')