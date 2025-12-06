from typing import List, Optional

import torch
from PIL import Image
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from app.schemas.common_schemas import (
    Box,
    GeneratedMask,
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

    async def generate_mask(
        self,
        image: Image.Image,
        points: Optional[List[MaskLabeledPoint]] = None,
        box: Optional[Box] = None,
    ) -> GeneratedMask:
        if points is None and box is None:
            raise ValueError("Either points or box must be provided")

        # Prepare inputs for tracker processor
        input_points = None
        input_labels = None
        input_boxes = None

        if points is not None:
            input_points = [[[[p.x, p.y] for p in points]]]
            input_labels = [[[p.label for p in points]]]

        if box is not None:
            input_boxes = [[[box.x_min, box.y_min, box.x_max, box.y_max]]]

        inputs = self._tracker_processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
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

    async def generate_masks(
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
