import os
from collections import defaultdict
import torch
import argparse
from SSR.datasets.thor import thor_datasets
import open3d as o3d

from SSR.training import trainer
from SSR.models.model_utils import run_network

import numpy as np
import yaml
import tqdm

from collections import OrderedDict

from imgviz import label_colormap


@torch.no_grad()
def render_fn(trainer, rays, chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            trainer.render_rays(rays[i:i+chunk])

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


# objects that respond the pickup action
PICKABLE_TO_COLOR = OrderedDict([
    ('Candle', (233, 102, 178)),
    ('SoapBottle', (168, 222, 137)),
    ('ToiletPaper', (162, 204, 152)),
    ('SoapBar', (43, 97, 155)),
    ('SprayBottle', (89, 126, 121)),
    ('TissueBox', (98, 43, 249)),
    ('DishSponge', (166, 58, 136)),
    ('PaperTowelRoll', (144, 173, 28)),
    ('Book', (43, 31, 148)),
    ('CreditCard', (56, 235, 12)),
    ('Dumbbell', (45, 57, 144)),
    ('Pen', (239, 130, 152)),
    ('Pencil', (177, 226, 23)),
    ('CellPhone', (227, 98, 136)),
    ('Laptop', (20, 107, 222)),
    ('CD', (65, 112, 172)),
    ('AlarmClock', (184, 20, 170)),
    ('Statue', (243, 75, 41)),
    ('Mug', (8, 94, 186)),
    ('Bowl', (209, 182, 193)),
    ('TableTopDecor', (126, 204, 158)),
    ('Box', (60, 252, 230)),
    ('RemoteControl', (187, 19, 208)),
    ('Vase', (83, 152, 69)),
    ('Watch', (242, 6, 88)),
    ('Newspaper', (19, 196, 2)),
    ('Plate', (188, 154, 128)),
    ('WateringCan', (147, 67, 249)),
    ('Fork', (54, 200, 25)),
    ('PepperShaker', (5, 204, 214)),
    ('Spoon', (235, 57, 90)),
    ('ButterKnife', (135, 147, 55)),
    ('Pot', (132, 237, 87)),
    ('SaltShaker', (36, 222, 26)),
    ('Cup', (35, 71, 130)),
    ('Spatula', (30, 98, 242)),
    ('WineBottle', (53, 130, 252)),
    ('Knife', (211, 157, 122)),
    ('Pan', (246, 212, 161)),
    ('Ladle', (174, 98, 216)),
    ('Egg', (240, 75, 163)),
    ('Kettle', (7, 83, 48)),
    ('Bottle', (64, 80, 115))])


# objects that respond to the open action
OPENABLE_TO_COLOR = OrderedDict([
    ('Drawer', (155, 30, 210)),
    ('Toilet', (21, 27, 163)),
    ('ShowerCurtain', (60, 12, 39)),
    ('ShowerDoor', (36, 253, 61)),
    ('Cabinet', (210, 149, 89)),
    ('Blinds', (214, 223, 197)),
    ('LaundryHamper', (35, 109, 26)),
    ('Safe', (198, 238, 160)),
    ('Microwave', (54, 96, 202)),
    ('Fridge', (91, 156, 207))])


# mapping from classes to colors for segmentation
CLASS_TO_COLOR = OrderedDict(
    [("OccupiedSpace", (243, 246, 208))]
    + list(PICKABLE_TO_COLOR.items())
    + list(OPENABLE_TO_COLOR.items()))


# number of semantic segmentation classes we process
NUM_CLASSES = len(CLASS_TO_COLOR)


def train():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default="./SSR/configs/thor_config.yaml", help='config file name.')

    parser.add_argument('--training_data_dir', type=str, required=True, help='Path to rendered data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to the directory saving training logs and ckpts.')

    parser.add_argument('--gpu', type=str, default="0", help='GPU IDs.')

    parser.add_argument('--return_top_p_occupied_voxels', type=float, default=0.5)
    parser.add_argument('--semantic_threshold', type=float, default=0.9)

    parser.add_argument('--voxel_size', type=float, default=0.01)

    parser.add_argument('--nb_neighbors', type=float, default=20)
    parser.add_argument('--std_ratio', type=float, default=1.)

    args = parser.parse_args()

    config_file_path = args.config_file

    # Read YAML file
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    if len(args.gpu) > 0:
        config["experiment"]["gpu"] = args.gpu

    print("Experiment GPU is {}.".format(config["experiment"]["gpu"]))
    trainer.select_gpus(config["experiment"]["gpu"])
        
    # Cast intrinsics to right types
    ssr_trainer = trainer.SSRTrainer(config)

    training_data_dir = args.training_data_dir
    save_dir = args.save_dir

    total_num = 199

    train_ids = list(range(4, total_num))
    test_ids = list(range(0, 4))

    thor_data_loader = thor_datasets.THORDatasetCache(
        data_dir=training_data_dir,
        train_ids=train_ids, test_ids=test_ids,
        img_h=config["experiment"]["height"],
        img_w=config["experiment"]["width"])

    ssr_trainer.set_params_thor()
    ssr_trainer.prepare_data_thor(thor_data_loader)

    ##########################

    # Create nerf model, init optimizer
    ssr_trainer.create_ssr()

    # Create rays in world coordinates
    ssr_trainer.init_rays()

    # load_ckpt into NeRF
    ckpt_path = os.path.join(save_dir, "checkpoints", "050000.ckpt")
    ckpt = torch.load(ckpt_path)

    ssr_trainer.ssr_net_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
    ssr_trainer.ssr_net_fine.load_state_dict(ckpt['network_fine_state_dict'])
    ssr_trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    ssr_trainer.training = False

    ssr_trainer.ssr_net_coarse.eval()
    ssr_trainer.ssr_net_fine.eval()

    ssr_trainer.return_top_p_occupied_voxels = args.return_top_p_occupied_voxels
    semantic_threshold = args.semantic_threshold

    point_cloud = []

    with torch.no_grad():
        for i, c2w in enumerate(tqdm.tqdm(ssr_trainer.rays_vis)):
            output_dict = ssr_trainer.render_rays(ssr_trainer.rays_vis[i])
            point_cloud.append(output_dict['occupied_pts_fine'].cpu().numpy())

    point_cloud = np.concatenate(point_cloud, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    pcd_coordinates = torch.as_tensor(
        pcd.points, dtype=torch.float32,
        device=f"cuda:{args.gpu}").unsqueeze(1)

    with torch.no_grad():

        chunk = 1024

        run_MLP_fn  =  lambda pts: run_network(inputs=pts, viewdirs=torch.zeros_like(pts).squeeze(1),
            fn=ssr_trainer.ssr_net_fine, embed_fn=ssr_trainer.embed_fn,
            embeddirs_fn=ssr_trainer.embeddirs_fn, netchunk=int(2048*128))

        def generate_segmentation(raw):

            sem_logits = raw[..., 4:]  # [N_rays, N_samples, num_class]
            sem_probabilities = torch.nn.functional.softmax(sem_logits, dim=-1)

            sem_labels = torch.argmax(sem_probabilities, dim=-1, keepdim=True)
            sem_labels = torch.where(torch.gather(sem_probabilities, -1, sem_labels) >
                                     semantic_threshold, sem_labels, torch.zeros_like(sem_labels))

            return torch.sigmoid(raw[..., :3]).cpu(), raw[..., 3].cpu(), sem_labels.cpu()

        outputs = [generate_segmentation(run_MLP_fn(pcd_coordinates[i: i+chunk]))
                   for i in range(0, pcd_coordinates.shape[0], chunk)]

        label_fine = torch.cat([x[2] for x in outputs], dim=0).view(-1).numpy()
        color_fine = torch.cat([x[0] for x in outputs], dim=0).view(-1, 3).numpy()

        pcd.colors = o3d.utility.Vector3dVector(color_fine)

    semantic_point_clouds = []

    for i in range(NUM_CLASSES):

        class_i_points = np.asarray(pcd.points)[(label_fine == i).nonzero()]
        class_i_colors = np.asarray(pcd.colors)[(label_fine == i).nonzero()]

        pcdi = o3d.geometry.PointCloud()

        pcdi.points = o3d.utility.Vector3dVector(class_i_points)
        pcdi.colors = o3d.utility.Vector3dVector(class_i_colors)

        pcdi = pcdi.voxel_down_sample(voxel_size=args.voxel_size)

        pcdi = pcdi.remove_statistical_outlier(
            nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)[0]

        semantic_point_clouds.append(pcdi)

    os.makedirs(os.path.join(args.save_dir, "point-clouds"), exist_ok=True)

    for i in range(NUM_CLASSES):

        o3d.io.write_point_cloud(os.path.join(
            args.save_dir, "point-clouds", f"class{i}.pcd"),
            semantic_point_clouds[i])


if __name__ == '__main__':
    train()
