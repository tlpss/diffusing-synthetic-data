import numpy as np
from PIL import Image

from dsd.cropped_diffusion_rendering import CropAndInpaintRenderer, CroppedRenderer
from dsd.diffusion_rendering import DiffusionRenderer, DiffusionRenderInputImages


class ThreeStageRenderer(DiffusionRenderer):
    def __init__(self, background_renderer: CropAndInpaintRenderer, object_renderer: CroppedRenderer):
        self.background_inpainter = background_renderer
        self.object_renderer = object_renderer

    def __call__(
        self,
        object_input_images: DiffusionRenderInputImages,
        surface_input_images: DiffusionRenderInputImages,
        object_prompt: str,
        surface_prompt: str,
    ):
        background_images = self.background_inpainter(surface_prompt, surface_prompt, surface_input_images)
        object_images = self.object_renderer(object_prompt, object_input_images)

        composed_images = []
        for object_image, background_image in zip(object_images, background_images):
            mask = Image.fromarray((object_input_images.mask * 255).astype(np.uint8))
            composed_image = Image.composite(object_image, background_image, mask)
            composed_images.append(composed_image)

        return composed_images
