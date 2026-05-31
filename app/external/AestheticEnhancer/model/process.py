import numpy as np
import random
from PIL import Image, ImageEnhance
import torch
from typing import Tuple

class ImageAugmenter:
    def __init__(self, img: Image):
        self.img = img

    # def adjust_vibrance(self, value=None):
    #     if value is None:
    #         # vibrance_value = random.uniform(-80, 80) # big range
    #         vibrance_value = random.uniform(-30, 30) 
    #     else:
    #         vibrance_value = value

    #     self.img = self.img.convert('HSV')
    #     hsv_array = np.array(self.img, dtype=np.float32)

    #     h, s, v = hsv_array[:, :, 0], hsv_array[:, :, 1], hsv_array[:, :, 2]
    #     intensity = np.mean(hsv_array, axis=2, keepdims=True)
    #     vibrance_factor = 1 + (vibrance_value / 100.0) * (1 - intensity / 255)

    #     s = s * vibrance_factor.squeeze()
    #     s = np.clip(s, 0, 255).astype(np.uint8)

    #     hsv_adjusted = np.stack([h, s, v], axis=2).astype(np.uint8)
    #     self.img = Image.fromarray(hsv_adjusted, 'HSV').convert('RGB')

    #     if vibrance_value > 0:
    #         return -round(vibrance_value, 1), f"Decrease vibrance by -{vibrance_value:.1f}"
    #     elif vibrance_value < 0:
    #         return abs(round(vibrance_value, 1)), f"Increase vibrance by {abs(vibrance_value):.1f}"
    #     else:
    #         return 0, "No vibrance adjustment"

    def adjust_saturation(self, value=None):
        if value is None:
            # saturation_value = random.uniform(-50, 80) # big range
            saturation_value = random.uniform(-30, 30) # small range
        else:
            saturation_value = value

        enhancer = ImageEnhance.Color(self.img)
        factor = (saturation_value + 100) / 100 # factor in [0.7, 1.3]
        self.img = enhancer.enhance(factor)
        
        adjusted_value = (1/factor) * 100 - 100

        if saturation_value > 0:
            return round(adjusted_value,1), f"Decrease saturation by {adjusted_value:.1f}"
        elif saturation_value < 0:
            return round(adjusted_value,1), f"Increase saturation by {adjusted_value:.1f}"
        else:
            return 0, "No saturation adjustment"

    def adjust_temperature(self, value=None):
        if value is None:
            # temp_value = random.uniform(-70, 70) # big range
            temp_value = random.uniform(-30, 30) # small range
        else:
            temp_value = value
        self.img = np.array(self.img).astype(np.float32)
        temp_adjust = -temp_value / 100.0

        self.img[:, :, 2] += temp_adjust * 50  # Blue channel
        self.img[:, :, 0] -= temp_adjust * 50  # Red channel
        self.img = np.clip(self.img, 0, 255).astype(np.uint8)
        self.img = Image.fromarray(self.img)

        if temp_value > 0:
            return round(-temp_value,1), f"Decrease temperature by -{temp_value:.1f}"
        if temp_value < 0:
            return round(abs(temp_value),1), f"Increase temperature by {abs(temp_value):.1f}"
        else:
            return 0, "No temperature adjustment"

    def adjust_tint(self, value=None):
        if value is None:
            # tint_value = random.uniform(-50, 50) # big range
            tint_value = random.uniform(-30, 30) # small range
        else:
            tint_value = value

        self.img = np.array(self.img).astype(np.float32)
        tint_adjust = tint_value / 100.0

        self.img[:, :, 1] += tint_adjust * 50        
        self.img = np.clip(self.img, 0, 255).astype(np.uint8)
        self.img = Image.fromarray(self.img)

        if tint_value > 0:
            return round(-tint_value,1), f"Decrease tint by -{tint_value:.1f}"
        elif tint_value < 0:
            return round(abs(tint_value),1), f"Increase tint by {abs(tint_value):.1f}"
        else:
            return 0, "No tint adjustment"

    def adjust_brightness(self, value=None):
        if value is None:
            # brightness_value = random.uniform(-33, 50) # big range
            brightness_value = random.uniform(-30, 30) # small range
        else:
            brightness_value = value

        enhancer = ImageEnhance.Brightness(self.img)
        factor = (brightness_value + 100) / 100
        self.img = enhancer.enhance(factor)
        
        adjusted_value = (1/factor) * 100 - 100

        if brightness_value > 0:
            return round(adjusted_value,1), f"Decrease brightness by {adjusted_value:.1f}"
        elif brightness_value < 0:
            return round(adjusted_value,1), f"Increase brightness by {adjusted_value:.1f}"
        else:
            return 0, "No brightness adjustment"

    def adjust_contrast(self, value=None):
        if value is None:
            # contrast_value = random.uniform(-50, 50) # big range
            contrast_value = random.uniform(-30, 30) # small range
        else:
            contrast_value = value

        enhancer = ImageEnhance.Contrast(self.img)
        factor = (contrast_value + 100) / 100
        self.img = enhancer.enhance(factor)
        
        adjusted_value = (1/factor) * 100 - 100

        if contrast_value > 0:
            return round(adjusted_value,1), f"Decrease contrast by {adjusted_value:.1f}"
        elif contrast_value < 0:
            return round(adjusted_value,1), f"Increase contrast by {adjusted_value:.1f}"
        else:
            return 0, "No contrast adjustment"

    def modify_randomly(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        descriptions = {}
        adjustments = {
            'saturation': 0,
            'brightness': 0,
            'tint': 0,
            # 'vibrance': 0,
            'temperature': 0,
            'contrast': 0
        }
        modifications = {
            'saturation': self.adjust_saturation,
            'brightness': self.adjust_brightness,
            'tint': self.adjust_tint,
            # 'vibrance': self.adjust_vibrance,
            'temperature': self.adjust_temperature,
            'contrast': self.adjust_contrast,
        }
        # chosen_mods = random.sample(list(modifications.keys()), random.randint(0, len(modifications)))
        for mod in modifications.keys():
            # if mod in chosen_mods:
            value, description = modifications[mod]()
            adjustments[mod] = value
            descriptions[mod] = description
            # else:
                # descriptions[mod] = f"No {mod} adjustment"

        # Create a tensor of adjustment values in the specified order
        # adjustment_values = torch.tensor([adjustments[key] for key in ['saturation', 'brightness', 'tint', 'vibrance', 'temperature', 'contrast']], dtype=torch.float32)

        return self.img, adjustments, ', '.join(descriptions.values())

    def modify_with_options(self, options, order=None):
        for factor in options:
            assert factor in ('saturation', 'tint', 'brightness', 'temperature', 'contrast'), f"{factor} is invalid"
        
        modifications = {
            'saturation': self.adjust_saturation,
            'brightness': self.adjust_brightness,
            'tint': self.adjust_tint,
            'temperature': self.adjust_temperature,
            'contrast': self.adjust_contrast,
        }

        if order is None:
            order = list(modifications.keys())

        for factor in order:
           modifications[factor](options[factor])

        return self.img 

    def generate_modifications(self, options) -> dict:
        """
        Generates a dictionary of modifications based on provided options.
        
        Args:
            options (dict): A dictionary where keys are factors (e.g., 'saturation', 'brightness')
                            and values are the adjustment values.
        
        Returns:
            dict: A dictionary where keys are factors and values are strings describing the adjustments.
        """
        descriptions = {}
        modifications = {
            'saturation': self.adjust_saturation,
            'brightness': self.adjust_brightness,
            'tint': self.adjust_tint,
            'temperature': self.adjust_temperature,
            'contrast': self.adjust_contrast,
        }
        
        # Apply each modification based on provided options
        for factor, adjustment_value in options.items():
            if abs(adjustment_value) > 5:
                value, description = modifications[factor](adjustment_value)
                descriptions[factor] = description
            else:
                descriptions[factor] = f"No {factor} adjustment"

        return descriptions

