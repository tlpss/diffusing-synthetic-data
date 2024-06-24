import queue
import random
import shutil
import time
from threading import Thread

import numpy as np
import torch
import tqdm

from dsd.diffusion_rendering import DiffusionRenderInputImages


class ImageSaver(Thread):
    """
    Thread class to save images generated by the diffusion model.
    This way the main thread can continue generating images while the images are saved in the background.
    """

    def __init__(self, image_queue, max_queue_size=20):
        super().__init__()
        self.image_queue = image_queue
        self.max_queue_size = max_queue_size
        self.daemon = True  # Set as daemon to avoid blocking program exit

    def run(self):
        while True:
            try:
                # Get image data from the queue with timeout to prevent blocking
                image, filename = self.image_queue.get(timeout=1)  # Set timeout to avoid deadlocks

                # Save the image
                from PIL import Image  # Import image library here for efficiency

                img = Image.fromarray(image)
                img.save(filename)
            except queue.Empty:
                # wait for 10s if the queue is empty
                time.sleep(10)
                # if the queue is still empty, break the loop
                if self.image_queue.empty():
                    break

        # Signal completion
        print("Image queue is empty, quiting")
        self.image_queue.task_done()


NOT_RERENDER_EXISTING = True


def generate_diffusion_renders(
    source_directory, target_directory, diffusion_renderers, prompts, num_prompts_per_scene=None
):
    image_saver = ImageSaver(queue.Queue(maxsize=20), max_queue_size=20)
    image_saver.start()

    rgb_image_paths = list(source_directory.glob("**/rgb.png"))
    image_dirs = [p.parent for p in rgb_image_paths]
    image_dirs = sorted(image_dirs)

    for renderer in tqdm.tqdm(diffusion_renderers):
        # fix seeds to make renders reproducible
        random.seed(2024)
        torch.manual_seed(2024)
        np.random.seed(2024)

        renderer, kwargs = renderer
        renderer = renderer(**kwargs)
        # disable NSFW filter
        renderer.pipe.safety_checker = None

        renderer.pipe.set_progress_bar_config(disable=True)
        for image_dir in tqdm.tqdm(image_dirs):
            relative_path_to_source_dir = image_dir.relative_to(source_directory)
            image_target_dir = target_directory / relative_path_to_source_dir

            if NOT_RERENDER_EXISTING and (image_target_dir / renderer.get_logging_name()).exists():
                continue

            image_target_dir.mkdir(parents=True, exist_ok=True)
            # copy the orignal images
            blender_image_target_dir = image_target_dir / "original"
            blender_image_target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in image_dir.glob("*"):
                shutil.copy(image_path, blender_image_target_dir)
            input_images = DiffusionRenderInputImages.from_render_dir(image_dir)

            prompts_to_render = random.sample(prompts, num_prompts_per_scene) if num_prompts_per_scene else prompts
            for prompt in prompts_to_render:
                renderer_image_target_dir = image_target_dir / renderer.get_logging_name()
                renderer_image_target_dir.mkdir(parents=True, exist_ok=True)
                output_images = renderer(prompt, input_images)
                for i, image in enumerate(output_images):

                    # path length is limited to avoid Filename too long error
                    length_limited_prompt = prompt[:50] + "...." if len(prompt) > 100 else prompt

                    # store a s jpg
                    image.save(renderer_image_target_dir / f"{length_limited_prompt}_{i}.jpg", quality=95)
                    # image_saver.image_queue.put((np.array(image), renderer_image_target_dir / f"{length_limited_prompt}_{i}.png"))

    image_saver.image_queue.join()  # Wait for the queue to be empty


def generate_crop_inpaint_diffusion_renders(
    source_directory, target_directory, diffusion_renderers, prompts, background_prompts, num_prompts_per_scene=None
):
    rgb_image_paths = list(source_directory.glob("**/rgb.png"))
    image_dirs = [p.parent for p in rgb_image_paths]
    image_dirs = sorted(image_dirs)

    for renderer in tqdm.tqdm(diffusion_renderers):
        # fix seeds to make renders reproducible
        random.seed(2024)
        torch.manual_seed(2024)
        np.random.seed(2024)

        # disable NSFW filter
        renderer.inpainter.pipe.safety_checker = None
        renderer.inpainter.pipe.set_progress_bar_config(disable=True)

        renderer.crop_renderer.renderer.pipe.safety_checker = None
        renderer.crop_renderer.renderer.pipe.set_progress_bar_config(disable=True)

        for image_dir in tqdm.tqdm(image_dirs, desc="crop-inpaint diffusion renderers"):
            relative_path_to_source_dir = image_dir.relative_to(source_directory)
            image_target_dir = target_directory / relative_path_to_source_dir
            image_target_dir.mkdir(parents=True, exist_ok=True)
            # copy the orignal images
            blender_image_target_dir = image_target_dir / "original"
            blender_image_target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in image_dir.glob("*"):
                # exclude blend file to reduce storage
                if image_path.suffix == ".blend":
                    continue
                shutil.copy(image_path, blender_image_target_dir)
            input_images = DiffusionRenderInputImages.from_render_dir(image_dir)

            prompts_to_render = random.sample(prompts, num_prompts_per_scene) if num_prompts_per_scene else prompts
            background_prompts_to_render = (
                random.sample(background_prompts, num_prompts_per_scene)
                if num_prompts_per_scene
                else background_prompts
            )
            for prompt, background_prompt in zip(prompts_to_render, background_prompts_to_render):
                renderer_image_target_dir = image_target_dir / renderer.get_logging_name()
                renderer_image_target_dir.mkdir(parents=True, exist_ok=True)
                output_images = renderer(prompt, background_prompt, input_images)
                for i, image in enumerate(output_images):
                    image.save(renderer_image_target_dir / f"{prompt}+{background_prompt}_{i}.png")
