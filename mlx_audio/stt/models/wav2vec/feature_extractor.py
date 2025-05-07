import json
import logging
import os
from collections import UserDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_audio.tts.utils import get_model_path

logger = logging.getLogger(__name__)


class TensorType(Enum):
    MX = "mx"
    NP = "np"


class BatchFeature(UserDict):
    def __init__(
        self,
        data=None,
        input_values: Any = None,
        attention_mask: Any = None,
        tensor_type: Union[str, TensorType] = TensorType.MX,
        **kwargs,
    ):
        super().__init__()
        if data:
            self.data.update(data)

        _input_values_key = "input_values"
        _attention_mask_key = "attention_mask"

        if input_values is not None:
            # Ensure input_values is a list of items
            if not (
                isinstance(input_values, list)
                and (
                    not input_values
                    or isinstance(input_values[0], (np.ndarray, mx.array, list, tuple))
                )
            ):
                self.data[_input_values_key] = [input_values]
            else:
                self.data[_input_values_key] = input_values

        if attention_mask is not None:
            # Ensure attention_mask is a list of items
            if not (
                isinstance(attention_mask, list)
                and (
                    not attention_mask
                    or isinstance(
                        attention_mask[0],
                        (np.ndarray, mx.array, list, tuple, type(None)),
                    )
                )
            ):
                self.data[_attention_mask_key] = [attention_mask]
            else:
                self.data[_attention_mask_key] = attention_mask

        if isinstance(tensor_type, str):
            self.tensor_type = TensorType(tensor_type)
        else:
            self.tensor_type = tensor_type

        # Update with any other kwargs passed
        self.data.update(kwargs)


class PaddingStrategy(Enum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


def load_json(path: os.PathLike) -> dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON file {path}: {e}")


class Wav2Vec2FeatureExtractor:
    r"""
    Constructs a Wav2Vec2 feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        feature_size (`int`, *optional*, defaults to 1):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding values.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean unit-variance normalize the input. Normalizing can help to significantly
            improve the performance for some models, *e.g.*,
            [wav2vec2-lv60](https://huggingface.co/models?search=lv60).
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~Wav2Vec2FeatureExtractor.__call__`] should return `attention_mask`.

            <Tip>

            Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
            `attention_mask`. For such models, `input_values` should simply be padded with 0 and no `attention_mask`
            should be passed.

            For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
            [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should be
            passed for batched inference.

            </Tip>"""

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
        self,
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        return_attention_mask=False,
        do_normalize=True,
        **kwargs,
    ):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.padding_side = kwargs.get("padding_side", "right")
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: List[np.ndarray],
        attention_mask: List[np.ndarray],
        padding_value: float = 0.0,
    ) -> List[np.ndarray]:
        """
        Every array in the list is normalized to have zero mean and unit variance
        """
        if attention_mask is not None:
            attention_mask = np.array(attention_mask, np.int32)
            normed_input_values = []

            for vector, length in zip(input_values, attention_mask.sum(-1)):
                normed_slice = (vector - vector[:length].mean()) / np.sqrt(
                    vector[:length].var() + 1e-7
                )
                if length < normed_slice.shape[0]:
                    normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [
                (x - x.mean()) / np.sqrt(x.var() + 1e-7) for x in input_values
            ]

        return normed_input_values

    def _truncate(
        self,
        processed_features: Union[dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        truncation: Optional[bool] = None,
    ):
        """
        Truncate inputs to predefined length or max length in the batch

        Args:
            processed_features(`Union[Dict[str, np.ndarray], BatchFeature]`):
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length (`int`, *optional*):
                maximum length of the returned list and optionally padding length (see below)
            pad_to_multiple_of (`int`, *optional*) :
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
        """
        if not truncation:
            return processed_features
        elif truncation and max_length is None:
            raise ValueError(
                "When setting ``truncation=True``, make sure that ``max_length`` is defined."
            )

        required_input = processed_features[self.model_input_names[0]]

        # find `max_length` that fits `pad_to_multiple_of`
        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_truncated = len(required_input) > max_length

        if needs_to_be_truncated:
            processed_features[self.model_input_names[0]] = processed_features[
                self.model_input_names[0]
            ][:max_length]
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features[
                    "attention_mask"
                ][:max_length]

        return processed_features

    def _get_padding_strategies(self, padding=False, max_length=None):
        """
        Find the correct padding strategy
        """

        # Get padding strategy
        if padding is not False:
            if padding is True:
                padding_strategy = (
                    PaddingStrategy.LONGEST
                )  # Default to pad to the longest sequence in the batch
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        # Set max length if needed
        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined"
                )

        # Test if we have a padding value
        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (
            self.padding_value is None
        ):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy

    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            list[BatchFeature],
            dict[str, BatchFeature],
            dict[str, list[BatchFeature]],
            list[dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, Any]] = None,
    ) -> BatchFeature:
        """
        Pad input values / input vectors or a batch of input values / input vectors up to predefined length or to the
        max sequence length in the batch.

        Padding side (left/right) padding values are defined at the feature extractor level (with `self.padding_side`,
        `self.padding_value`)

        <Tip>

        If the `processed_features` passed are dictionary of numpy arrays, PyTorch tensors or TensorFlow tensors, the
        result will use the same type unless you provide a different tensor type with `return_tensors`. In the case of
        PyTorch tensors, you will lose the specific device of your tensors however.

        </Tip>

        Args:
            processed_features ([`BatchFeature`], list of [`BatchFeature`], `Dict[str, List[float]]`, `Dict[str, List[List[float]]` or `List[Dict[str, List[float]]]`):
                Processed inputs. Can represent one input ([`BatchFeature`] or `Dict[str, List[float]]`) or a batch of
                input values / vectors (list of [`BatchFeature`], *Dict[str, List[List[float]]]* or *List[Dict[str,
                List[float]]]*) so you can use this method during preprocessing as well as in a PyTorch Dataloader
                collate function.

                Instead of `List[float]` you can have tensors (numpy arrays, PyTorch tensors or TensorFlow tensors),
                see the note above for the return type.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'mx'`: Return MXNet `mx.ndarray` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(processed_features, (list, tuple)) and isinstance(
            processed_features[0], (dict, BatchFeature)
        ):
            processed_features = {
                key: [example[key] for example in processed_features]
                for key in processed_features[0].keys()
            }

        # The model's main input name, usually `input_values`, has be passed for padding
        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                "You should supply an instance of `transformers.BatchFeature` or list of `transformers.BatchFeature`"
                f" to this method that includes {self.model_input_names[0]}, but you provided"
                f" {list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask
            if return_attention_mask is not None
            else self.return_attention_mask
        )

        if len(required_input) == 0:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        # If we have PyTorch/TF tensors or lists as inputs, we cast them as Numpy arrays
        # and rebuild them afterwards if no return_tensors is specified
        # Note that we lose the specific device the tensor may be on for PyTorch

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]

        if return_tensors is None:
            if isinstance(first_element, mx.array):
                return_tensors = "mx"
            elif isinstance(first_element, (int, float, list, tuple, np.ndarray)):
                return_tensors = "np"
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

        for key, value in processed_features.items():
            if isinstance(value[0], (int, float)):
                processed_features[key] = np.array(value)
            else:
                processed_features[key] = [np.array(v) for v in value]

        # Convert padding_strategy in PaddingStrategy
        padding_strategy = self._get_padding_strategies(
            padding=padding, max_length=max_length
        )

        required_input = processed_features[self.model_input_names[0]]

        batch_size = len(required_input)
        if not all(len(v) == batch_size for v in processed_features.values()):
            raise ValueError(
                "Some items in the output dictionary have a different batch size than others."
            )

        truncated_inputs = []
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in processed_features.items()}
            # truncation
            inputs_slice = self._truncate(
                inputs,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=truncation,
            )
            truncated_inputs.append(inputs_slice)

        if padding_strategy == PaddingStrategy.LONGEST:
            # make sure that `max_length` cannot be longer than the longest truncated length
            max_length = max(
                len(input_slice[self.model_input_names[0]])
                for input_slice in truncated_inputs
            )
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            # padding
            outputs = self._pad(
                truncated_inputs[i],
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                if value.dtype is np.dtype(np.float64):
                    value = value.astype(np.float32)
                batch_outputs[key].append(value)

        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        processed_features: Union[dict[str, np.ndarray], BatchFeature],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            processed_features (`Union[Dict[str, np.ndarray], BatchFeature]`):
                Dictionary of input values (`np.ndarray[float]`) / input vectors (`List[np.ndarray[float]]`) or batch
                of inputs values (`List[np.ndarray[int]]`) / input vectors (`List[np.ndarray[int]]`)
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see below)
            padding_strategy (`PaddingStrategy`, *optional*, default to `PaddingStrategy.DO_NOT_PAD`):
                PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The feature_extractor padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of (`int`, *optional*):
                Integer if set will pad the sequence to a multiple of the provided value. This is especially useful to
                enable the use of Tensor Core on NVIDIA hardware with compute capability `>= 7.5` (Volta), or on TPUs
                which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Set to False to avoid returning attention mask (default: set to model specifics)
        """
        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if (
            max_length is not None
            and pad_to_multiple_of is not None
            and (max_length % pad_to_multiple_of != 0)
        ):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD
            and len(required_input) < max_length
        )

        if return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = np.ones(
                len(required_input), dtype=np.int32
            )

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (0, difference)
                    )
                padding_shape = (
                    ((0, difference), (0, 0))
                    if self.feature_size > 1
                    else (0, difference)
                )
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    constant_values=self.padding_value,
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (difference, 0)
                    )
                padding_shape = (
                    ((difference, 0), (0, 0))
                    if self.feature_size > 1
                    else (difference, 0)
                )
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input,
                    padding_shape,
                    "constant",
                    constant_values=self.padding_value,
                )
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return processed_features

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, Any]] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                Wav2Vec2 models that have set `config.feat_extract_norm == "group"`, such as
                [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), have **not** been trained using
                `attention_mask`. For such models, `input_values` should simply be padded with 0 and no
                `attention_mask` should be passed.

                For Wav2Vec2 models that have set `config.feat_extract_norm == "layer"`, such as
                [wav2vec2-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self), `attention_mask` should
                be passed for batched inference.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'mx'`: Return MXNet `mx.ndarray` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding_value (`float`, *optional*, defaults to 0.0):
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                f"It is strongly recommended to pass the `sampling_rate` argument to `{self.__class__.__name__}()`. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = (
            isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        )
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(
                f"Only mono-channel audio is supported for input to {self}"
            )
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_values": raw_speech})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        # convert input values to correct format
        input_values = padded_inputs["input_values"]
        if not isinstance(input_values[0], np.ndarray):
            padded_inputs["input_values"] = [
                np.asarray(array, dtype=np.float32) for array in input_values
            ]
        elif (
            not isinstance(input_values, np.ndarray)
            and isinstance(input_values[0], np.ndarray)
            and input_values[0].dtype is np.dtype(np.float64)
        ):
            padded_inputs["input_values"] = [
                array.astype(np.float32) for array in input_values
            ]
        elif isinstance(input_values, np.ndarray) and input_values.dtype is np.dtype(
            np.float64
        ):
            padded_inputs["input_values"] = input_values.astype(np.float32)

        # convert attention_mask to correct format
        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None:
            padded_inputs["attention_mask"] = [
                np.asarray(array, dtype=np.int32) for array in attention_mask
            ]

        # zero-mean and unit-variance normalization
        if self.do_normalize:
            attention_mask = (
                attention_mask
                if self._get_padding_strategies(padding, max_length=max_length)
                is not PaddingStrategy.DO_NOT_PAD
                else None
            )
            padded_inputs["input_values"] = self.zero_mean_unit_var_norm(
                padded_inputs["input_values"],
                attention_mask=attention_mask,
                padding_value=self.padding_value,
            )

        if return_tensors is not None:
            for k, v in padded_inputs.items():
                if return_tensors == "mx":
                    # Convert to numpy array first if it's not already one
                    if isinstance(v, list):
                        v = np.array(v)
                    padded_inputs[k] = mx.array(v)
                elif return_tensors == "np":
                    padded_inputs[k] = np.array(v)
                else:
                    raise ValueError(f"Invalid return_tensors: {return_tensors}")
        return padded_inputs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        file_name: str = "preprocessor_config.json",
        revision: str = "main",
        **kwargs,
    ):
        if isinstance(pretrained_model_name_or_path, str):
            path = get_model_path(pretrained_model_name_or_path)
        else:
            path = pretrained_model_name_or_path

        if not (path / file_name).exists():
            raise FileNotFoundError(f"File {file_name} not found in {path}")

        feature_extractor_dict = load_json(path / file_name)

        return cls.from_dict(feature_extractor_dict, **kwargs)

    @classmethod
    def from_dict(
        cls, feature_extractor_dict: dict[str, Any], **kwargs
    ) -> "Wav2Vec2FeatureExtractor":
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if key in feature_extractor_dict:
                feature_extractor_dict[key] = value
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        feature_extractor = cls(**feature_extractor_dict)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor
