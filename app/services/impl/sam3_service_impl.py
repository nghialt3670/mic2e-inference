from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image, ImageDraw
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from app.schemas.common_schemas import (
    Box,
    GeneratedMask,
    MaskLabeledBox,
    MaskLabeledPoint,
)
from app.services.sam3_service import Sam3Service


class Sam3ServiceImpl(Sam3Service):
    def __init__(
        self,
        model: Sam3Model,
        processor: Sam3Processor,
        tracker_model: Sam3TrackerModel,
        tracker_processor: Sam3TrackerProcessor,
    ):
        self._model = model
        self._processor = processor
        self._tracker_model = tracker_model
        self._tracker_processor = tracker_processor

    async def generate_mask_by_points(
        self, image: Image.Image, points: List[MaskLabeledPoint]
    ) -> GeneratedMask:
        input_points = [[[[p.x, p.y] for p in points]]]
        input_labels = [[[p.label for p in points]]]
        inputs = self._tracker_processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self._tracker_model.device)

        with torch.no_grad():
            outputs = self._tracker_model(**inputs)

        scores = outputs.iou_scores[0, 0].cpu().tolist()
        best_index = scores.index(max(scores))
        mask = self._tracker_processor.post_process_masks(
            outputs.pred_masks, original_sizes=inputs["original_sizes"]
        )[0][0][best_index]

        mask_image = Image.fromarray(mask.cpu().numpy().astype("uint8") * 255, mode="L")

        return GeneratedMask(image=mask_image, score=scores[best_index])

    async def generate_mask_by_box(self, image: Image.Image, box: Box) -> GeneratedMask:
        inputs = self._processor(
            images=image,
            input_boxes=[[[box.x_min, box.y_min, box.x_max, box.y_max]]],
            input_boxes_labels=[[1]],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs, target_sizes=inputs["original_sizes"].tolist()
        )

        masks = results[0].get("masks").detach().cpu().numpy()
        scores = results[0].get("scores").detach().cpu().tolist()
        best_index = scores.index(max(scores))
        mask = masks[best_index]
        mask_image = Image.fromarray(mask.astype("uint8") * 255, mode="L")

        return GeneratedMask(image=mask_image, score=scores[best_index])

    async def generate_masks_by_text(
        self, image: Image.Image, text: str
    ) -> List[GeneratedMask]:
        inputs = self._processor(images=image, return_tensors="pt", text=text).to(
            self._model.device
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_instance_segmentation(
            outputs, target_sizes=inputs["original_sizes"].tolist()
        )

        masks = results[0].get("masks")
        scores = [float(score) for score in results[0].get("scores")]
        mask_images = [
            Image.fromarray(mask.cpu().numpy().astype("uint8") * 255, mode="L")
            for mask in masks
        ]

        return [
            GeneratedMask(image=mask_image, score=score)
            for mask_image, score in zip(mask_images, scores)
        ]
