import argparse
import open3d
import glob
import os
import numpy as np


from imgviz import label_colormap
from collections import OrderedDict, defaultdict


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    args = parser.parse_args()

    label_colour_map = label_colormap()

    files = glob.glob(os.path.join(args.save_dir, "*.pcd"))

    files = sorted(files, key=lambda f: int(os.path.basename(f)[5:-4]))

    clouds = [open3d.io.read_point_cloud(f) for f in files]

    bboxes = []

    class_to_max_size = defaultdict(int)
    class_to_detections = defaultdict(list)

    for f, c in zip(files, clouds):

        object_id = int(os.path.basename(f)[5:-4])

        if object_id == 0:
            continue  # this is the background class

        object_color = label_colour_map[object_id].astype(np.float32) / 255.0

        num_points = np.asarray(c.colors).shape[0]

        c.colors = open3d.utility.Vector3dVector(
            np.tile(object_color, (num_points, 1)))

        labels = np.array(c.cluster_dbscan(
            eps=2 * args.voxel_size, min_points=10))

        indices = np.nonzero(labels >= 0)

        for l in np.unique(labels[indices]):

            pcdi = open3d.geometry.PointCloud()

            indices_l = np.nonzero(labels == l)

            pcdi.colors = open3d.utility.Vector3dVector(
                np.asarray(c.colors)[indices_l])
            pcdi.points = open3d.utility.Vector3dVector(
                np.asarray(c.points)[indices_l])

            aabb = pcdi.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)  # Red

            bboxes.append(aabb)

            print(list(CLASS_TO_SIZE.keys())[object_id],
                  indices_l[0].size)

            class_to_max_size[object_id] = max(
                class_to_max_size[object_id], indices_l[0].size)

            class_to_detections[object_id].append((pcdi, aabb))

    for object_id, detections in class_to_detections.items():

        class_to_detections[object_id] = []

        for pcdi, aabb in detections:

            if (np.asarray(pcdi.colors).shape[0] >
                    class_to_max_size[object_id] * 0.2):

                class_to_detections[object_id].append((pcdi, aabb))

    open3d.visualization.draw_geometries([
        pi for k, v in class_to_detections.items() for vi in v for pi in vi])


if __name__ == '__main__':
    main()
