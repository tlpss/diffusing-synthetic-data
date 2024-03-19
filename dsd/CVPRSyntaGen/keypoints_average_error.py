import torch
from keypoint_detection.data.coco_dataset import COCOKeypointsDataset
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from keypoint_detection.utils.load_checkpoints import get_model_from_wandb_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_average_error_for_dataset(
    model: KeypointDetector, dataset_json_path, channel_config: list[list[str]], detect_only_visible_keypoints
):
    dataset = COCOKeypointsDataset(
        dataset_json_path,
        keypoint_channel_configuration=channel_config,
        detect_only_visible_keypoints=detect_only_visible_keypoints,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    model.eval()
    model.cuda()

    errors = [[] for _ in range(len(channel_config))]
    for image, keypoints in tqdm(dataloader):
        image = image.cuda()
        heatmaps = model(image)
        keypoints = keypoints
        predicted_keypoints = get_keypoints_from_heatmap_batch_maxpool(heatmaps, max_keypoints=1)[0]
        for i in range(len(channel_config)):
            if len(predicted_keypoints[i]) == 0:
                print("no keypoints found")
                # if no keypoints are found, we assume the center of the image as the predicted keypoint
                # predicted_keypoints[i] = torch.ones(1, 2) * image.shape[-1] / 2
                continue
            if len(keypoints[i]) == 0:
                print("no GT keypoints found")
                continue
            kp = torch.tensor(predicted_keypoints[i][0], dtype=torch.float32)
            gt_kp = torch.tensor(keypoints[i][0], dtype=torch.float32)
            l2_error = torch.norm(kp - gt_kp)
            errors[i].append(l2_error.item())

    average_errors = [sum(errors[i]) / len(errors[i]) for i in range(len(channel_config))]
    for i in range(len(channel_config)):
        print(f"Average error for channel {channel_config[i]}: {average_errors[i]}")
    print(f"Average error: {sum(average_errors)/len(average_errors)}")

    mae_dict = {}
    for i in range(len(channel_config)):
        channel_name = "" + "-".join(channel_config[i])
        mae_dict[channel_name] = average_errors[i]
    return mae_dict


if __name__ == "__main__":
    from dsd import DATA_DIR

    wandb_checkpoint = "tlips/dsd-mugs-cvpr/model-s4yfql9w:v0"  # controlnet-depth.
    wandb_checkpoint = "tlips/dsd-mugs-cvpr/model-1myo185a:v0"  # lab-mugs
    wandb_checkpoint = "tlips/dsd-mugs-cvpr/model-2esztwoc:v0"  # SD2-depth
    dataset_json_path = DATA_DIR / "real" / "mugs" / "lab-mugs_resized_512x512" / "lab-mugs_train.json"
    model = get_model_from_wandb_checkpoint(wandb_checkpoint).cuda()
    avg_errors = calculate_average_error_for_dataset(
        model, dataset_json_path, [["bottom"], ["handle"], ["top"]], False
    )
    print(avg_errors)
