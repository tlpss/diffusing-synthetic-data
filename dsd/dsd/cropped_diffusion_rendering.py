import copy
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from dsd.diffusion_rendering import DiffusionRenderer, DiffusionRenderInputImages


@dataclass
class Bbox:
    min_x: int
    max_x: int
    min_y: int
    max_y: int


def crop_images_to_bbox(input_images: DiffusionRenderInputImages, bbox: Bbox) -> DiffusionRenderInputImages:
    """in-place cropping of all input images to the object mask"""
    min_x, max_x, min_y, max_y = bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y
    input_images.rgb_image = cv2.resize(
        input_images.rgb_image[min_y:max_y, min_x:max_x], input_images.rgb_image.shape[:2][::-1]
    )
    input_images.depth_image = cv2.resize(
        input_images.depth_image[min_y:max_y, min_x:max_x], input_images.depth_image.shape[:2][::-1]
    )
    input_images.mask = cv2.resize(input_images.mask[min_y:max_y, min_x:max_x], input_images.mask.shape[:2][::-1])

    return input_images


def get_bbox_from_mask(mask: np.ndarray) -> Bbox:
    boolean_mask = mask > 0.5
    non_zero_indices = np.where(boolean_mask == True)  # noqa
    min_x = non_zero_indices[1].min()
    max_x = non_zero_indices[1].max()
    min_y = non_zero_indices[0].min()
    max_y = non_zero_indices[0].max()

    return Bbox(min_x, max_x, min_y, max_y)


def add_margin_to_bbox(bbox: Bbox, margin: int, mask: np.ndarray) -> Bbox:
    min_x = max(0, bbox.min_x - margin)
    max_x = min(mask.shape[1], bbox.max_x + margin)
    min_y = max(0, bbox.min_y - margin)
    max_y = min(mask.shape[0], bbox.max_y + margin)
    return Bbox(min_x, max_x, min_y, max_y)


class ObjectCroppedDiffusionRenderInputImages(DiffusionRenderInputImages):
    def __init__(self, input_images: DiffusionRenderInputImages, bbox_padding: int = 10):
        self.original_input_images = input_images
        self.original_bbox = get_bbox_from_mask(input_images.mask)
        self.original_bbox = add_margin_to_bbox(self.original_bbox, bbox_padding, input_images.mask)

        cropped_input_images = copy.deepcopy(input_images)
        cropped_input_images.rgb_image = np.copy(input_images.rgb_image)
        cropped_input_images.depth_image = np.copy(input_images.depth_image)
        cropped_input_images.mask = np.copy(input_images.mask)

        cropped_input_images = crop_images_to_bbox(cropped_input_images, self.original_bbox)
        super().__init__(cropped_input_images.rgb_image, cropped_input_images.depth_image, cropped_input_images.mask)


class CroppedRenderer(DiffusionRenderer):
    def __init__(self, renderer: DiffusionRenderer, bbox_padding: int = 10, only_change_mask: bool = True):
        self.renderer = renderer
        self.bbox_margin = bbox_padding
        self.only_change_mask = only_change_mask

    def __call__(self, prompt, input_images: DiffusionRenderInputImages, **kwargs):
        cropped_input_images = ObjectCroppedDiffusionRenderInputImages(input_images)
        cropped_results = self.renderer(prompt, cropped_input_images, **kwargs)
        cropped_results_numpy = [np.array(image).astype(np.float32) / 255.0 for image in cropped_results]
        # return cropped_results
        results_numpy = [np.copy(input_images.rgb_image) for _ in cropped_results_numpy]
        # paste in the results in the bbox and then overlay with the original mask
        for i, cropped_result in enumerate(cropped_results_numpy):
            results_numpy[i][
                cropped_input_images.original_bbox.min_y : cropped_input_images.original_bbox.max_y,
                cropped_input_images.original_bbox.min_x : cropped_input_images.original_bbox.max_x,
            ] = cv2.resize(
                cropped_result,
                (
                    cropped_input_images.original_bbox.max_x - cropped_input_images.original_bbox.min_x,
                    cropped_input_images.original_bbox.max_y - cropped_input_images.original_bbox.min_y,
                ),
            )
            if self.only_change_mask:
                # mask the original image, so that only the object mask is changed
                # note that this is not the same as using inpainting, because the diffusion model never gets to see the original image
                results_numpy[i][input_images.mask < 0.5] = input_images.rgb_image[input_images.mask < 0.5]
        results = [Image.fromarray((image * 255).astype(np.uint8)) for image in results_numpy]
        return results

    def get_logging_name(self) -> str:
        return f"Cropped:{self.renderer.get_logging_name()},margin={self.bbox_margin},only_change_mask={self.only_change_mask}"


class CropAndInpaintRenderer(DiffusionRenderer):
    def __init__(
        self, crop_renderer: CroppedRenderer, inpainter: DiffusionRenderer, mask_dilation_iterations: int = 1
    ):
        self.crop_renderer = crop_renderer
        self.inpainter = inpainter
        self.mask_dilation_iterations = mask_dilation_iterations

    def __call__(self, prompt: str, background_prompt: str, input_images: DiffusionRenderInputImages, **kwargs):
        # print("prompt", prompt)
        # print("background_prompt", background_prompt)

        stage_1_results = self.crop_renderer(prompt, input_images, **kwargs)
        stage_2_inputs = DiffusionRenderInputImages(
            np.copy(input_images.rgb_image), np.copy(input_images.depth_image), np.copy(input_images.mask)
        )
        original_mask = np.copy(input_images.mask)

        # invert the mask to inpaint the background
        # but first dilate it slightly to limit 'blurring' by the diffusion inpaint model
        # which tries to make the transition between the mask and the non-mask smooth
        stage_2_inputs.mask = cv2.dilate(
            stage_2_inputs.mask, np.ones((5, 5), np.uint8), iterations=self.mask_dilation_iterations
        )
        stage_2_inputs.mask = 1 - stage_2_inputs.mask
        # duplicate the input image to create one for each stage 1 result
        stage_2_inputs = [copy.deepcopy(stage_2_inputs) for _ in stage_1_results]

        stage_2_results = []
        # add the stage 1 results to the stage 2 results
        # stage_2_results = [x for x in stage_1_results]
        for i, result in enumerate(stage_1_results):
            stage_2_inputs[i].rgb_image = np.array(result).astype(np.float32) / 255.0
            stage_2_result = self.inpainter(background_prompt, stage_2_inputs[i], **kwargs)
            for j, image in enumerate(stage_2_result):
                # paste the original image back in the mask
                # to reduce any artifacts from the inpainting
                image = np.array(image).astype(np.float32) / 255.0
                image[original_mask > 0.5] = stage_2_inputs[i].rgb_image[original_mask > 0.5]
                stage_2_result[j] = Image.fromarray((image * 255).astype(np.uint8))
            stage_2_results.extend(stage_2_result)

        return stage_2_results

    def get_logging_name(self) -> str:
        return f"2stage:crop={self.crop_renderer.get_logging_name()},inp={self.inpainter.get_logging_name()},dilation={self.mask_dilation_iterations}"


# def crop_and_diffuse(crop_renderer: CroppedRenderer, inpainter: DiffusionRenderer, input_images: DiffusionRenderInputImages, object_prompt, background_prompt,**kwargs):
#     stage_1_results = crop_renderer(object_prompt, input_images, **kwargs)
#     stage_2_input = DiffusionRenderInputImages(np.copy(input_images.rgb_image), np.copy(input_images.depth_image), np.copy(input_images.mask))
#     # invert the mask to inpaint the background
#     stage_2_input.mask = 1 - stage_2_input.mask
#     # duplicate the input image to create one for each stage 1 result
#     stage_2_input = [copy.deepcopy(stage_2_input) for _ in stage_1_results]
#     # set the input images to the stage 1 results
#     stage_2_results = []
#     for i, result in enumerate(stage_1_results):
#         stage_2_input[i].rgb_image = np.array(result).astype(np.float32)/255.0
#         stage_2_result = inpainter(background_prompt, stage_2_input[i], **kwargs)
#         # for each element, overlay the original image non-masked on the inpainted image

#         stage_2_results.extend(stage_2_result)

#     return stage_1_results, stage_2_results
