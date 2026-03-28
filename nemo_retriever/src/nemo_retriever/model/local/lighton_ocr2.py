# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image

from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from nemo_retriever.utils.hf_model_registry import get_hf_revision
from ..model import BaseModel, RunMode


class LightOnOCR2(BaseModel):
    """
    LightOn OCR 2 local wrapper.

    This wrapper loads ``lightonai/LightOn-OCR-2`` from Hugging Face and
    runs image-to-text generation for document parsing.  A custom
    ``model_path`` can be supplied to point at a different checkpoint or a
    locally downloaded copy of the model.

    Usage::

        model = LightOnOCR2()
        text = model.invoke(pil_image)

    For remote inference against a LightOn-compatible chat-completions
    endpoint (OpenAI API format) use ``lighton_ocr2_page_elements`` with an
    ``invoke_url`` instead of instantiating this class.
    """

    def __init__(
        self,
        model_path: str = "lightonai/LightOn-OCR-2",
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self._model_path = model_path
        self._device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        hf_cache_dir = configure_global_hf_cache_base(hf_cache_dir)
        _revision = get_hf_revision(self._model_path)

        self._model = AutoModel.from_pretrained(
            self._model_path,
            revision=_revision,
            trust_remote_code=True,
            torch_dtype=self._dtype,
            cache_dir=hf_cache_dir,
        ).to(self._device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            revision=_revision,
            cache_dir=hf_cache_dir,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_path,
            revision=_revision,
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
        )

    def preprocess(self, input_data: Union[torch.Tensor, np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """
        Normalize supported input formats to a RGB PIL image.
        """
        if isinstance(input_data, Image.Image):
            return input_data.convert("RGB")

        if isinstance(input_data, (str, Path)):
            return Image.open(Path(input_data)).convert("RGB")

        if isinstance(input_data, torch.Tensor):
            x = input_data.detach().cpu()
            if x.ndim == 4:
                if int(x.shape[0]) != 1:
                    raise ValueError(f"Expected batch size 1 tensor, got shape {tuple(x.shape)}")
                x = x[0]
            if x.ndim != 3:
                raise ValueError(f"Expected CHW/HWC tensor, got shape {tuple(x.shape)}")
            if int(x.shape[0]) in (1, 3):
                x = x.permute(1, 2, 0).contiguous()
            if x.dtype.is_floating_point:
                max_v = float(x.max().item()) if x.numel() else 1.0
                if max_v <= 1.5:
                    x = x * 255.0
            arr = x.clamp(0, 255).to(torch.uint8).numpy()
            return Image.fromarray(arr).convert("RGB")

        if isinstance(input_data, np.ndarray):
            arr = input_data
            if arr.ndim == 4:
                if int(arr.shape[0]) != 1:
                    raise ValueError(f"Expected batch size 1 array, got shape {arr.shape}")
                arr = arr[0]
            if arr.ndim != 3:
                raise ValueError(f"Expected HWC/CHW array, got shape {arr.shape}")
            if int(arr.shape[0]) in (1, 3) and int(arr.shape[-1]) not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if np.issubdtype(arr.dtype, np.floating):
                max_v = float(arr.max()) if arr.size else 1.0
                if max_v <= 1.5:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")

        raise TypeError(f"Unsupported input type for LightOn OCR 2: {type(input_data)!r}")

    def invoke(
        self,
        input_data: Union[torch.Tensor, np.ndarray, Image.Image, str, Path],
        prompt: Optional[str] = None,
    ) -> str:
        """
        Run local LightOn OCR 2 inference and return the decoded model text.

        Parameters
        ----------
        input_data:
            Document image in any supported format (PIL Image, file path,
            torch.Tensor CHW/BCHW, or numpy HWC/CHW array).
        prompt:
            Optional text prompt passed to the processor.  When ``None`` the
            processor's default prompt (if any) is used.
        """
        image = self.preprocess(input_data)

        processor_kwargs: dict = {"images": [image], "return_tensors": "pt"}
        if prompt is not None:
            processor_kwargs["text"] = prompt

        inputs = self._processor(**processor_kwargs).to(self._device)

        with torch.inference_mode():
            outputs = self._model.generate(**inputs)

        decoded = self._processor.batch_decode(outputs, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    def __call__(
        self,
        input_data: Union[torch.Tensor, np.ndarray, Image.Image, str, Path],
        prompt: Optional[str] = None,
    ) -> str:
        return self.invoke(input_data, prompt=prompt)

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "LightOn-OCR-2"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "document-parse"

    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        """Input schema for the model."""
        return {
            "type": "image",
            "format": "RGB",
            "supported_inputs": ["PIL.Image", "path", "torch.Tensor", "np.ndarray"],
            "description": "Document image for OCR parsing into markdown text.",
        }

    @property
    def output(self) -> Any:
        """Output schema for the model."""
        return {
            "type": "text",
            "format": "string",
            "description": "Generated structured text from LightOn OCR 2.",
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return 1
