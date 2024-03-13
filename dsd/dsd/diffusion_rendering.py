from __future__ import annotations

import os

# enable opencv exr
# https://github.com/opencv/opencv/issues/21326
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pathlib
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from controlnet_aux import CannyDetector, PidiNetDetector
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image


@dataclass
class DiffusionRenderInputImages:
    rgb_image: np.ndarray
    """ (H,W,3) RGB image as numpy array in range [0,1]"""
    depth_image: np.ndarray
    """ (H,W) absolute depth map as numpy array"""
    mask: np.ndarray
    """ (H,W) mask as numpy array, 1.0 for object pixels, 0.0 for background pixels"""

    @classmethod
    def from_render_dir(cls, render_dir: str):
        render_dir = pathlib.Path(render_dir)
        rgb_image = Image.open(render_dir / "rgb.png")
        rgb_image = np.array(rgb_image)
        rgb_image = rgb_image.astype(np.float32) / 255.0

        # read the exr file using opencv
        depth_image = cv2.imread(str(render_dir / "depth_map.exr"), cv2.IMREAD_UNCHANGED)
        depth_image = depth_image[..., 0]
        # controlnet actually uses relative, inverted depth
        # cf. Midas model https://pytorch.org/hub/intelisl_midas_v2/
        # depth_image = invert_and_normalize_depth_image(depth_image)

        mask = Image.open(render_dir / "segmentation.png")
        mask = np.array(mask)
        mask = mask.astype(np.float32) / 255.0
        return cls(rgb_image, depth_image, mask)

    def get_rgb_image_torch(self):
        rgb_image_torch = np.copy(self.rgb_image)
        rgb_image_torch = torch.from_numpy(self.rgb_image).permute(2, 0, 1).unsqueeze(0)
        return rgb_image_torch

    def get_inverted_depth_image_torch(self):
        depth_image = np.copy(self.depth_image)
        inverted_depth_image = _invert_and_normalize_depth_image(depth_image)
        inverted_depth_image = torch.from_numpy(inverted_depth_image)
        # duplicate 3 channels
        inverted_depth_image = inverted_depth_image.repeat(3, 1, 1).unsqueeze(0)
        return inverted_depth_image

    def get_inverted_depth_image(self):
        return _invert_and_normalize_depth_image(self.depth_image)

    def get_normal_map(self):
        inverted_depth_image = self.get_inverted_depth_image()
        normal_map = _get_controlnet_like_normals_from_depth_image(inverted_depth_image)
        return normal_map

    def get_normal_map_torch(self):
        normal_map = self.get_normal_map()
        normal_map = torch.from_numpy(normal_map).permute(2, 0, 1).unsqueeze(0)
        return normal_map

    def get_mask_torch(self):
        mask_torch = torch.from_numpy(self.mask).unsqueeze(0).unsqueeze(0)
        return mask_torch


def _get_controlnet_like_normals_from_depth_image(depth_image: np.ndarray, sobel_kernel_size: int = 7) -> np.ndarray:
    assert len(depth_image.shape) == 2
    assert depth_image.dtype == np.float32

    x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)
    z = np.ones_like(depth_image)
    normals = np.stack([x, y, z], axis=-1)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = (normals + 1) / 2  # (-1,1) to (0,1)

    return normals


def _invert_and_normalize_depth_image(depth_image):
    depth_image = 1 - depth_image
    depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    return depth_image


def get_canny_edges_from_image(image: Image.Image, low_threshold=20, high_threshold=100) -> Image.Image:
    image = np.array(image)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = image * 255
        image = image.astype(np.uint8)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = Image.fromarray(image)
    return image


# SD2 benefits from negative prompts: https://stable-diffusion-art.com/how-to-use-negative-prompts/#Universal_negative_prompt
SDV2_NEGATIVE_PROMPT = "ugly, tiling, out of frame, disfigured, deformed, body out of frame, bad anatomy, watermark,signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"


class DiffusionRenderer:
    def __init__(self, num_images_per_prompt=4, num_inference_steps=50):
        self.generator = torch.manual_seed(1)
        self.num_images_per_prompt = num_images_per_prompt
        self.num_inference_steps = num_inference_steps
        self.input_resolution = (512, 512)  # (w,h)

    def __call__(self, prompt: str, input_images: DiffusionRenderInputImages, **kwargs) -> list[Image.Image]:
        raise NotImplementedError

    def get_logging_name(self):
        return self.__class__.__name__


class SD2InpaintingRenderer(DiffusionRenderer):
    """
    Stable Diffusion 2 Inpainting model
    https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

    """

    def __init__(self, num_images_per_prompt=4, num_inference_steps=50, strength=1.0):
        super().__init__(num_images_per_prompt, num_inference_steps)
        self.strength = strength
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")

    def __call__(self, prompt, input_images, **kwargs):
        rgb_image = input_images.get_rgb_image_torch()
        mask = input_images.get_mask_torch()

        output_dict = self.pipe(
            prompt=prompt,
            negative_prompt=SDV2_NEGATIVE_PROMPT,
            image=rgb_image,
            mask_image=mask,
            num_images_per_prompt=self.num_images_per_prompt,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            generator=self.generator,
        )
        images = output_dict.images
        return images

    def get_logging_name(self):
        return f"{self.__class__.__name__}_strength={self.strength}"


class SD2RegularCheckpointInpaintRenderer(DiffusionRenderer):
    """
    SD 2.0 base checkpoint used for inpainting

    """

    def __init__(self, num_images_per_prompt=4, num_inference_steps=50, strength=1.0):
        super().__init__(num_images_per_prompt, num_inference_steps)
        self.strength = strength
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            torch_dtype=torch.float16,
        ).to("cuda")

    def __call__(self, prompt, input_images, **kwargs):
        rgb_image = input_images.get_rgb_image_torch()
        mask = input_images.get_mask_torch()

        output_dict = self.pipe(
            prompt=prompt,
            negative_prompt=SDV2_NEGATIVE_PROMPT,
            image=rgb_image,
            mask_image=mask,
            num_images_per_prompt=self.num_images_per_prompt,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            generator=self.generator,
        )
        images = output_dict.images
        return images

    def get_logging_name(self):
        return f"{self.__class__.__name__}_strength={self.strength}"


class SD15InpaintingRenderer(DiffusionRenderer):
    def __init__(self, num_images_per_prompt=4, num_inference_steps=50, strength=1.0):
        super().__init__(num_images_per_prompt, num_inference_steps)
        self.strength = strength
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")

    def __call__(self, prompt, input_images, **kwargs):
        rgb_image = input_images.get_rgb_image_torch()
        mask = input_images.get_mask_torch()

        output_dict = self.pipe(
            prompt=prompt,
            image=rgb_image,
            mask_image=mask,
            num_images_per_prompt=self.num_images_per_prompt,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            generator=self.generator,
        )
        images = output_dict.images
        return images

    def get_logging_name(self):
        return f"{self.__class__.__name__}_strength={self.strength}"


class SD2FromDepthRenderer(DiffusionRenderer):
    """
    Stable Diffusion 2 + native depth conditioning
    https://huggingface.co/stabilityai/stable-diffusion-2-depth


    use the GenAug settings for this model.
    """

    def __init__(self, num_images_per_prompt=4, num_inference_steps=50, strength=0.9):
        super().__init__(num_images_per_prompt, num_inference_steps)
        self.strength = strength

        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
        ).to("cuda")

    def __call__(self, prompt, input_images, **kwargs):
        # depth_image = input_images.get_inverted_depth_image_torch()
        # use settings as in GenAug: no GT depth

        # rgb_image = input_images.get_rgb_image_torch()

        # TOOD: figure out why directly passing the torch tensor does not work?
        rgb_image = Image.fromarray((input_images.rgb_image * 255).astype(np.uint8))

        output_dict = self.pipe(
            prompt=prompt,
            image=rgb_image,
            negative_prompt=SDV2_NEGATIVE_PROMPT,
            num_images_per_prompt=self.num_images_per_prompt,
            num_inference_steps=self.num_inference_steps,
            strength=self.strength,
            generator=self.generator,
        )
        images = output_dict.images
        return images


class ControlNetRenderer(DiffusionRenderer):
    DEFAULT_CONTROLNET_CONDITIONING_SCALE = 1.5
    DEFAULT_STRENGTH = 1.0

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = DEFAULT_STRENGTH,
        use_img2img_pipeline: bool = True,
    ):
        super().__init__(num_images_per_prompt, num_inference_steps)
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.strength = strength
        self.pipe: DiffusionPipeline = None
        self.use_img2img_pipeline = use_img2img_pipeline

    def get_logging_name(self):
        return f"{self.__class__.__name__}_ccs={self.controlnet_conditioning_scale}"  # _{'img2img' if self.use_img2img_pipeline else 'txt'}"

    def __call__(self, prompt, input_images, **kwargs):
        # hacky way to check if this pipeline is an Img2Img pipeline or a regular txt2img pipeline

        # the reason for this is that I have the feeling that even with strenght=1.0, there is still quite some influence from the
        # initial image, which reduces diversity and complexity of the generated images?
        # TODO: measure this.
        if self.use_img2img_pipeline:
            img = self.preprocess_rgb_image(input_images)
            control = self.get_control_image(input_images)
            output_dict = self.pipe(
                prompt=prompt,
                image=img,
                control_image=control,
                num_images_per_prompt=self.num_images_per_prompt,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                generator=self.generator,
            )
        else:
            control = self.get_control_image(input_images)
            output_dict = self.pipe(
                prompt=prompt,
                image=control,  # image = control image!
                num_images_per_prompt=self.num_images_per_prompt,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                num_inference_steps=self.num_inference_steps,
                generator=self.generator,
            )
        images = output_dict.images
        return images

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        raise NotImplementedError

    def preprocess_rgb_image(self, input_images: DiffusionRenderInputImages) -> torch.Tensor:

        # if np.isclose(self.strength, 1.0):
        #     return torch.randn((1,4,64,64))
        rgb_image = input_images.get_rgb_image_torch()
        if rgb_image.shape[:2] != self.input_resolution:
            rgb_image = torch.nn.functional.interpolate(rgb_image, size=self.input_resolution, mode="bicubic")
        return rgb_image


class SDXLControlNetFromDepthRenderer(ControlNetRenderer):
    """
    SD-XL + Controlnet, depth conditioned.
    https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetImg2ImgPipeline
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_conditioning_scale, strength)
        self.input_resolution = (1024, 1024)  # SD-XL is trained on 1024x1024 images

        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        depth_image = input_images.get_inverted_depth_image_torch()
        # resize the images to 1024x1024
        depth_image = torch.nn.functional.interpolate(depth_image, size=(1024, 1024), mode="bicubic")
        # return torch.randn(1, 3, 1024, 1024)
        return depth_image

    def __call__(self, prompt, input_images, **kwargs):

        images = super().__call__(prompt, input_images, **kwargs)
        images = [image.resize((512, 512)) for image in images]
        return images


class SDXLControlNetTXTFromDepthRenderer(ControlNetRenderer):
    """
    SD-XL + Controlnet, depth conditioned.
    https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetImg2ImgPipeline
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):
        super().__init__(
            num_images_per_prompt,
            num_inference_steps,
            controlnet_conditioning_scale,
            strength,
            use_img2img_pipeline=False,
        )
        self.input_resolution = (1024, 1024)  # SD-XL is trained on 1024x1024 images

        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        depth_image = input_images.get_inverted_depth_image_torch()
        # resize the images to 1024x1024
        depth_image = torch.nn.functional.interpolate(depth_image, size=(1024, 1024), mode="bicubic")
        # return torch.randn(1, 3, 1024, 1024)
        return depth_image

    def __call__(self, prompt, input_images, **kwargs):

        images = super().__call__(prompt, input_images, **kwargs)
        images = [image.resize((512, 512)) for image in images]
        return images


class SDXLControlNetFromCannyRenderer(ControlNetRenderer):
    """
    SD-XL + Controlnet, canny conditioned.
    https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl#diffusers.StableDiffusionXLControlNetImg2ImgPipeline
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_conditioning_scale, strength)
        self.input_resolution = (1024, 1024)  # SD-XL is trained on 1024x1024 images

        self.controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=self.controlnet,
            vae=self.vae,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
        self.pipe.enable_model_cpu_offload()

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        rgb_image = Image.fromarray((input_images.rgb_image * 255).astype(np.uint8))
        rgb_image = rgb_image.resize((1024, 1024))
        control_image = get_canny_edges_from_image(rgb_image)
        # duplicate 3 channels
        control_image = np.array(control_image)
        control_image = np.concatenate(
            [control_image[..., None], control_image[..., None], control_image[..., None]], axis=-1
        )
        control_image = Image.fromarray(control_image)
        return control_image

    def preprocess_rgb_image(self, input_images: DiffusionRenderInputImages) -> torch.Tensor:
        rgb_image = Image.fromarray((input_images.rgb_image * 255).astype(np.uint8))
        rgb_image = rgb_image.resize((1024, 1024))
        return rgb_image

    def __call__(self, prompt, input_images, **kwargs):

        images = super().__call__(prompt, input_images, **kwargs)
        images = [image.resize((512, 512)) for image in images]
        return images


class SD15RealisticCheckpointControlNetFromDepthRenderer(ControlNetRenderer):
    """
    SD 1.5 finetuned checkpoint for increased Realism + controlnet trained on inverse depth.
    checkpoint from https://civitai.com/models/4201?modelVersionId=245598

    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):

        # TODO: configure which controlnet and corresponding diffusion model to use?
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_conditioning_scale, strength)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        ).to("cuda")
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V6.0_B1_noVAE", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to("cuda")

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        return input_images.get_inverted_depth_image_torch()


class SD15RealisticCheckpointControlNetTXTFromDepthRenderer(ControlNetRenderer):
    """
    SD 1.5 finetuned checkpoint for increased Realism + controlnet trained on inverse depth.
    checkpoint from https://civitai.com/models/4201?modelVersionId=245598

    txt-to-image pipeline

    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):

        # TODO: configure which controlnet and corresponding diffusion model to use?
        super().__init__(
            num_images_per_prompt,
            num_inference_steps,
            controlnet_conditioning_scale,
            strength,
            use_img2img_pipeline=False,
        )
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V6.0_B1_noVAE", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to("cuda")

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        return input_images.get_inverted_depth_image_torch()


class ControlNetFromDepthRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on 'inverse' depth
    https://huggingface.co/lllyasviel/sd-controlnet-depth
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):

        # TODO: configure which controlnet and corresponding diffusion model to use?
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_conditioning_scale, strength)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to("cuda")

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        return input_images.get_inverted_depth_image_torch()


class ControlNetTXTFromDepthRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on 'inverse' depth
    https://huggingface.co/lllyasviel/sd-controlnet-depth
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):

        # TODO: configure which controlnet and corresponding diffusion model to use?
        super().__init__(
            num_images_per_prompt,
            num_inference_steps,
            controlnet_conditioning_scale,
            strength,
            use_img2img_pipeline=False,
        )
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to("cuda")

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        return input_images.get_inverted_depth_image_torch()


class ControlnetfromHEDRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on HED edges
    https://huggingface.co/lllyasviel/sd-controlnet-hed
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_conditioning_scale, strength)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_softedge", torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to("cuda")
        self.processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        self.use_rgb_for_control = True

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        control_image = input_images.rgb_image if self.use_rgb_for_control else input_images.get_inverted_depth_image()
        control_image = Image.fromarray((control_image * 255).astype(np.uint8))
        control_image = self.processor(control_image)
        return control_image


class ControlNetFromCannyRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on canny edges
    https://huggingface.co/lllyasviel/sd-controlnet-canny
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_conditioning_scale, strength)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to("cuda")
        self.processor = CannyDetector()
        self.use_rgb_for_control = True

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        control_image = input_images.rgb_image if self.use_rgb_for_control else input_images.get_inverted_depth_image()
        control_image = get_canny_edges_from_image(control_image)
        return control_image


class ControlNetFromNormalsRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on 'normal' maps

    https://huggingface.co/lllyasviel/sd-controlnet-normal
    """

    def __init__(
        self,
        num_images_per_prompt: int = 4,
        num_inference_steps: int = 50,
        controlnet_conditioning_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        strength: float = ControlNetRenderer.DEFAULT_STRENGTH,
    ):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_conditioning_scale, strength)

        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        ).to("cuda")

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        control_image = input_images.get_normal_map_torch()

        return control_image
