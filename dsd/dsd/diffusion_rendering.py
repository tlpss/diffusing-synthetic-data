from __future__ import annotations

import os 
# enable opencv exr
# https://github.com/opencv/opencv/issues/21326
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 

from dataclasses import dataclass
import pathlib
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline
import requests
import torch
from PIL import Image
import numpy as np 
from controlnet_aux import PidiNetDetector, CannyDetector



@dataclass
class DiffusionRenderInputImages:
    rgb_image: np.ndarray
    """ (H,W,3) RGB image as numpy array in range [0,1]"""
    depth_image: np.ndarray
    """ (H,W) absolute depth map as numpy array"""

    @classmethod 
    def from_render_dir(cls, render_dir: str):
        render_dir = pathlib.Path(render_dir)
        rgb_image = Image.open(render_dir / "rgb.png")
        rgb_image = np.array(rgb_image)
        rgb_image = rgb_image.astype(np.float32) / 255.0

        # read the exr file using opencv
        depth_image = cv2.imread(str(render_dir / "depth_map.exr"), cv2.IMREAD_UNCHANGED)
        depth_image = depth_image[...,0]
        # controlnet actually uses relative, inverted depth 
        # cf. Midas model https://pytorch.org/hub/intelisl_midas_v2/
        #depth_image = invert_and_normalize_depth_image(depth_image)
        return cls(rgb_image, depth_image)


    def get_rgb_image_torch(self):
        rgb_image_torch = torch.from_numpy(self.rgb_image).permute(2,0,1).unsqueeze(0)
        return rgb_image_torch
    
    def get_inverted_depth_image_torch(self):
        inverted_depth_image = _invert_and_normalize_depth_image(self.depth_image)
        inverted_depth_image =  torch.from_numpy(inverted_depth_image)
        # duplicate 3 channels
        inverted_depth_image = inverted_depth_image.repeat(3,1,1).unsqueeze(0)
        return inverted_depth_image
    
    def get_inverted_depth_image(self):
        return _invert_and_normalize_depth_image(self.depth_image)

    def get_normal_map(self):
        inverted_depth_image = self.get_inverted_depth_image()
        normal_map = _get_controlnet_like_normals_from_depth_image(inverted_depth_image)
        return normal_map

    def get_normal_map_torch(self):
        normal_map = self.get_normal_map()
        normal_map = torch.from_numpy(normal_map).permute(2,0,1).unsqueeze(0)
        return normal_map
    
def _get_controlnet_like_normals_from_depth_image(depth_image: np.ndarray, sobel_kernel_size:int = 7) -> np.ndarray:
    assert len(depth_image.shape) == 2
    assert depth_image.dtype == np.float32

    x = cv2.Sobel(depth_image, cv2.CV_32F, 1, 0, ksize=sobel_kernel_size)
    y = cv2.Sobel(depth_image, cv2.CV_32F, 0, 1, ksize=sobel_kernel_size)
    z = np.ones_like(depth_image)
    normals = np.stack([x,y,z], axis=-1)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = (normals + 1) / 2 # (-1,1) to (0,1)

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



    
class DiffusionRenderer:
    def __init__(self, num_images_per_prompt = 4, num_inference_steps = 50):
        self.generator = torch.manual_seed(1)
        self.num_images_per_prompt = num_images_per_prompt
        self.num_inference_steps = num_inference_steps
        self.input_resolution = (512, 512) # (w,h)


    def __call__(self,prompt: str, input_images: DiffusionRenderInputImages, **kwargs) -> list[Image.Image]:
        raise NotImplementedError



class SDInpaintingRenderer(DiffusionRenderer):
    pass
class SDFromDepthRenderer(DiffusionRenderer):
    pass


    
class ControlNetRenderer(DiffusionRenderer):
    DEFAULT_CONTROLNET_GUIDANCE_SCALE = 1.0
    DEFAULT_STRENGTH = 1.0
    def __init__(self, num_images_per_prompt: int = 4, num_inference_steps: int = 50, controlnet_guidance_scale: float =DEFAULT_CONTROLNET_GUIDANCE_SCALE, strength: float = DEFAULT_STRENGTH):
        super().__init__(num_images_per_prompt, num_inference_steps)
        self.controlnet_guidance_scale = controlnet_guidance_scale
        self.strength = strength


    def get_control_image(self, input_images: DiffusionRenderInputImages):
        raise NotImplementedError

class SDXLControlNetFromDepthRenderer(ControlNetRenderer):
    pass 

class ControlNetFromDepthRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on 'inverse' depth
    https://huggingface.co/lllyasviel/sd-controlnet-depth
    """
    def __init__(self, num_images_per_prompt: int = 4, num_inference_steps: int = 50, controlnet_guidance_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_GUIDANCE_SCALE, strength: float = ControlNetRenderer.DEFAULT_STRENGTH):

        # TODO: configure which controlnet and corresponding diffusion model to use?
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_guidance_scale,strength)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16).to("cuda")
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                                             controlnet=self.controlnet, torch_dtype=torch.float16).to("cuda")
    def __call__(self,prompt, input_images, **kwargs):
        # TODO: better to resize in the pipeline?
        assert input_images.rgb_image.shape[:2] == self.input_resolution
        output_dict = self.pipe(prompt=prompt,
                            image = input_images.get_rgb_image_torch(),
                            control_image = self.get_control_image(input_images),
                            num_images_per_prompt = self.num_images_per_prompt,
                            controlnet_guidance_scale = self.controlnet_guidance_scale,
                            strength = self.strength, 
                            num_inference_steps = self.num_inference_steps,
                            generator= self.generator,
                            )
        images = output_dict.images
        return images

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        return input_images.get_inverted_depth_image_torch()

class ControlnetfromHEDRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on HED edges
    https://huggingface.co/lllyasviel/sd-controlnet-hed
    """
    def __init__(self, num_images_per_prompt: int = 4, num_inference_steps: int = 50, controlnet_guidance_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_GUIDANCE_SCALE, strength: float = ControlNetRenderer.DEFAULT_STRENGTH):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_guidance_scale, strength)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_softedge", torch_dtype=torch.float16).to("cuda")
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                                             controlnet=self.controlnet, torch_dtype=torch.float16).to("cuda")
        self.processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        self.use_rgb_for_control = False

    def __call__(self,prompt, input_images, **kwargs):

        output_dict = self.pipe(prompt=prompt,
                            image = input_images.rgb_image,
                            control_image = self.get_control_image(input_images),
                            num_images_per_prompt = self.num_images_per_prompt,
                            controlnet_condition_scale = self.controlnet_guidance_scale,
                            strength = self.strength,
                            num_inference_steps = self.num_inference_steps,
                            generator= self.generator,
                            **kwargs
                            )
        images = output_dict.images
        return images
    
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
    def __init__(self, num_images_per_prompt: int = 4, num_inference_steps: int = 50, controlnet_guidance_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_GUIDANCE_SCALE, strength: float = ControlNetRenderer.DEFAULT_STRENGTH):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_guidance_scale, strength)
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to("cuda")
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                                             controlnet=self.controlnet, torch_dtype=torch.float16).to("cuda")
        self.processor = CannyDetector()
        self.use_rgb_for_control = True

    def __call__(self,prompt, input_images, **kwargs):
        output_dict = self.pipe(prompt=prompt,
                    image = input_images.rgb_image,
                    control_image = self.get_control_image(input_images),
                    num_images_per_prompt = self.num_images_per_prompt,
                    controlnet_condition_scale = self.controlnet_guidance_scale,
                    num_inference_steps = self.num_inference_steps,
                    strength = self.strength,
                    generator= self.generator,
                    **kwargs
                    )
        images = output_dict.images
        return images

    def get_control_image(self, input_images: DiffusionRenderInputImages):
        control_image = input_images.rgb_image if self.use_rgb_for_control else input_images.get_inverted_depth_image()
        control_image = get_canny_edges_from_image(control_image)
        return control_image
    
class ControlNetFromNormalsRenderer(ControlNetRenderer):
    """
    SD1.5 + controlnet 1.1 trained on 'normal' maps

    https://huggingface.co/lllyasviel/sd-controlnet-normal
    """
    def __init__(self, num_images_per_prompt: int = 4, num_inference_steps: int = 50, controlnet_guidance_scale: float = ControlNetRenderer.DEFAULT_CONTROLNET_GUIDANCE_SCALE, strength: float = ControlNetRenderer.DEFAULT_STRENGTH):
        super().__init__(num_images_per_prompt, num_inference_steps, controlnet_guidance_scale, strength)
        
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal", torch_dtype=torch.float16).to("cuda")
        
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                                                             controlnet=self.controlnet, torch_dtype=torch.float16).to("cuda")
    def __call__(self,prompt, input_images, **kwargs):
        output_dict = self.pipe(prompt=prompt,
                    image = input_images.rgb_image,
                    control_image = self.get_control_image(input_images),
                    num_images_per_prompt = self.num_images_per_prompt,
                    controlnet_condition_scale = self.controlnet_guidance_scale,
                    num_inference_steps = self.num_inference_steps,
                    strength = self.strength,
                    generator= self.generator,
                    **kwargs
                    )
        images = output_dict.images
        return images
    
    def get_control_image(self, input_images: DiffusionRenderInputImages):
        control_image = input_images.get_normal_map_torch()

        return control_image

