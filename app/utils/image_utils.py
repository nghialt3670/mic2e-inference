import zipfile
from io import BytesIO
from typing import List

from fastapi import UploadFile
from PIL import Image


async def read_upload_file_as_image(upload_file: UploadFile) -> Image.Image:
    image_bytes = await upload_file.read()
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def zip_images_to_bytes(images: List[Image.Image], filenames: List[str]) -> BytesIO:
    zip_bytes = BytesIO()
    with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for image, filename in zip(images, filenames):
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes.seek(0)
            zip_file.writestr(filename, image_bytes.read())
    zip_bytes.seek(0)
    return zip_bytes


def save_image_to_bytes(image: Image.Image) -> BytesIO:
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes
