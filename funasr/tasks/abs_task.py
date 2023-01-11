import argparse
import logging
import sys
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Callable,Dict,List,Optional,Sequence,Tuple,Union
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from funasr.train.class_choices import ClassChoices
from funasr.train.trainer import Trainer
from funasr.train.abs_espnet_model import AbsESPnetModel
from funasr.datasets.iterable_dataset_modelscope import IterableESPnetDatasetModelScope, IterableESPnetBytesModelScope

class AbsTask(ABC):
    num_optimizers: int = 1
    trainer = Trainer
    class_choices_list: List[ClassChoices] = []

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def build_streaming_iterator_modelscope(
            cls,
            data_path_and_name_and_type,
            preprocess_fn,
            collate_fn,
            key_file: str = None,
            batch_size: int = 1,
            dtype: str = np.float32,
            num_workers: int = 1,
            allow_variable_data_keys: bool = False,
            ngpu: int = 0,
            inference: bool = False,
            sample_rate: Union[dict, int] = 16000
    ) -> DataLoader:
        """Build DataLoader using iterable dataset"""
        assert check_argument_types()
        # For backward compatibility for pytorch DataLoader
        if collate_fn is not None:
            kwargs = dict(collate_fn=collate_fn);print('collate_fn');input('abs_task.py 1773')
        else:
            kwargs = {}

        audio_data = data_path_and_name_and_type[0];print('audio_data',audio_data);print('abs_task.py',1777);input('')
        if isinstance(audio_data, bytes):
            print('1779 isinstance(audio_data, bytes)');input('')
            dataset = IterableESPnetBytesModelScope(
                data_path_and_name_and_type,
                float_dtype=dtype,
                preprocess=preprocess_fn,
                key_file=key_file,
                sample_rate=sample_rate
            )
        else:
            print('1788 dataset = IterableESPnetDatasetModelScope');input('')
            dataset = IterableESPnetDatasetModelScope(
                data_path_and_name_and_type,
                float_dtype=dtype,
                preprocess=preprocess_fn,
                key_file=key_file,
                sample_rate=sample_rate
            )

        if dataset.apply_utt2category:
            kwargs.update(batch_size=1)
        else:
            kwargs.update(batch_size=batch_size)

        return DataLoader( dataset=dataset, pin_memory=ngpu > 0, num_workers=num_workers, **kwargs )

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
            cls,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            device: str = "cpu",
    ) -> Tuple[AbsESPnetModel, argparse.Namespace]:

        print('abs_task.py 85 config_file',config_file)
        print('abs_task.py 86 model_file',model_file)
        # input('')
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        # device='cuda'
        model.to(device)
        if model_file is not None:
            if device == "cuda":
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                device = f"cuda:{torch.cuda.current_device()}"
            model.load_state_dict(torch.load(model_file, map_location=device))

        return model, args
