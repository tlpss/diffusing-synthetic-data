import json
import pathlib

import torch
import tqdm
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor, set_seed


def caption_coco_dataset(coco_dataset_json_path, n_captions_per_image=1):
    set_seed(2024)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map=0
    )

    def ask_blip2(question, img):
        inputs = processor(img, question, return_tensors="pt").to("cuda:0", torch.float16)
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
        return processor.decode(outputs[0], skip_special_tokens=True).strip()

    caption_dict = {}
    coco_dataset = json.load(open(coco_dataset_json_path))
    coco_dataset = CocoKeypointsDataset(**coco_dataset)
    for image in tqdm.tqdm(coco_dataset.images):
        image_path = pathlib.Path(coco_dataset_json_path).parent / image.file_name
        img = Image.open(image_path)
        question = "Question: Describe this image, including a detailed description of the environment. Answer:"
        captions = []
        for i in range(n_captions_per_image):
            caption = ask_blip2(question, img)
            captions.append(caption)
        caption_dict[image.file_name] = captions
    return caption_dict


if __name__ == "__main__":
    from dsd import DATA_DIR

    # generate +- 3000 captions for each category from the real train split.
    # coco_dataset_json_path = DATA_DIR / "real" / "mugs" / "dsd-mugs-robot" / "annotations_train.json"
    # caption_dict = caption_coco_dataset(coco_dataset_json_path,2)
    # json.dump(caption_dict, open("mug-captions.json", "w"))
    # coco_dataset_json_path = DATA_DIR / "real" / "shoes" / "dsd-shoes-robot-kpcentercropped" / "annotations_train.json"
    # caption_dict = caption_coco_dataset(coco_dataset_json_path,2)
    # json.dump(caption_dict, open("shoe-captions.json", "w"))

    coco_dataset_json_path = (
        DATA_DIR / "real" / "tshirts" / "tshirts-train.val-kpcentercropped" / "annotations_train.json"
    )
    caption_dict = caption_coco_dataset(coco_dataset_json_path, 20)
    json.dump(caption_dict, open("tshirt-captions.json", "w"))
