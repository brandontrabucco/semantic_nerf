import argparse
import open3d
import glob
import os
import numpy as np
import pickle as pkl


from imgviz import label_colormap
from collections import OrderedDict, defaultdict


# objects that respond the pickup action
PICKABLE_TO_RATIO = OrderedDict([
    ('Candle', 0.2),
    ('SoapBottle', 0.2),
    ('ToiletPaper', 0.2),
    ('SoapBar', 0.2),
    ('SprayBottle', 0.2),
    ('TissueBox', 0.2),
    ('DishSponge', 0.2),
    ('PaperTowelRoll', 0.2),
    ('Book', 0.2),
    ('CreditCard', 0.2),
    ('Dumbbell', 0.2),
    ('Pen', 0.2),
    ('Pencil', 0.2),
    ('CellPhone', 0.2),
    ('Laptop', 0.2),
    ('CD', 0.2),
    ('AlarmClock', 0.2),
    ('Statue', 0.2),
    ('Mug', 0.2),
    ('Bowl', 0.2),
    ('TableTopDecor', 0.2),
    ('Box', 0.2),
    ('RemoteControl', 0.2),
    ('Vase', 0.2),
    ('Watch', 0.2),
    ('Newspaper', 0.2),
    ('Plate', 0.2),
    ('WateringCan', 0.2),
    ('Fork', 0.2),
    ('PepperShaker', 0.2),
    ('Spoon', 0.2),
    ('ButterKnife', 0.2),
    ('Pot', 0.2),
    ('SaltShaker', 0.2),
    ('Cup', 0.2),
    ('Spatula', 0.2),
    ('WineBottle', 0.2),
    ('Knife', 0.2),
    ('Pan', 0.2),
    ('Ladle', 0.2),
    ('Egg', 0.2),
    ('Kettle', 0.2),
    ('Bottle', 0.2)])


# objects that respond to the open action
OPENABLE_TO_RATIO = OrderedDict([
    ('Drawer', 0.2),
    ('Toilet', 0.2),
    ('ShowerCurtain', 0.2),
    ('ShowerDoor', 0.2),
    ('Cabinet', 0.2),
    ('Blinds', 0.2),
    ('LaundryHamper', 0.2),
    ('Safe', 0.2),
    ('Microwave', 0.2),
    ('Fridge', 0.2)])


# mapping from classes to colors for segmentation
CLASS_TO_RATIO = OrderedDict(
    [("OccupiedSpace", 0.2)]
    + list(PICKABLE_TO_RATIO.items())
    + list(OPENABLE_TO_RATIO.items()))


# number of semantic segmentation classes we process
NUM_CLASSES = len(CLASS_TO_RATIO)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    label_colour_map = label_colormap()

    files = glob.glob(os.path.join(args.save_dir, "*.pcd"))
    files = [f for f in files if "object" not in f]
    files = sorted(files, key=lambda f: int(os.path.basename(f)[5:-4]))

    clouds = [open3d.io.read_point_cloud(f) for f in files]

    bboxes = []

    class_to_max_size = defaultdict(int)
    class_to_detections = defaultdict(list)

    for f, c in zip(files, clouds):

        object_id = int(os.path.basename(f)[5:-4])

        if object_id == 0:
            continue  # this is the background class

        labels = np.array(c.cluster_dbscan(
            eps=2 * args.voxel_size, min_points=0))

        indices = np.nonzero(labels >= 0)

        for l in np.unique(labels[indices]):

            pcdi = open3d.geometry.PointCloud()

            indices_l = np.nonzero(labels == l)

            pcdi.colors = open3d.utility.Vector3dVector(
                np.asarray(c.colors)[indices_l])
            pcdi.points = open3d.utility.Vector3dVector(
                np.asarray(c.points)[indices_l])

            aabb = pcdi.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)

            bboxes.append(aabb)

            class_to_max_size[object_id] = max(
                class_to_max_size[object_id], indices_l[0].size)

            class_to_detections[object_id].append((pcdi, aabb))

    class_to_bounding_boxes = {k: [] for k in class_to_detections.keys()}

    for object_id, detections in class_to_detections.items():

        class_to_detections[object_id] = []

        for pcdi, aabb in detections:

            min_size_ratio = list(CLASS_TO_RATIO.values())[object_id]

            if (np.asarray(pcdi.colors).shape[0] >
                    class_to_max_size[object_id] * min_size_ratio):

                num_instances = len(class_to_detections[object_id])

                open3d.io.write_point_cloud(os.path.join(
                    args.save_dir, f"class{object_id}_object{num_instances}.pcd"), pcdi)

                class_to_detections[object_id].append((pcdi, aabb))

                class_to_bounding_boxes[object_id].append(np.concatenate([
                    aabb.get_min_bound().reshape([3]),
                    aabb.get_extent().reshape([3])], axis=0))

    for key, value in class_to_bounding_boxes.items():
        class_to_bounding_boxes[key] = np.stack(value, axis=0)

    with open(os.path.join(
            args.save_dir, "detections.pkl"), "wb") as f:
        pkl.dump(class_to_bounding_boxes, f)

    if args.show:

        open3d.visualization.draw_geometries([
            pi for k, v in class_to_detections
            .items() for vi in v for pi in vi])


if __name__ == '__main__':
    main()
