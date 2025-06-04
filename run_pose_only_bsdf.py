# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Modified for HRT1 by Jishnu Jaykumar Padalunkal

"""
removed 3D reconstruction
contains only feature correspondence and pose estimation
"""


from bundlesdf import *
import argparse
import os, sys, glob, shutil
from PIL import Image as PILImage


code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
# from segmentation_utils import Segmenter

def remove_jpg_dirs(root_dir):
    logging.info(f"Removing directories containing .jpg in {root_dir}")
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and ".jpg" in item:
            shutil.rmtree(item_path)

def remove_ply_and_config(root_dir):
    logging.info(f"Removing .ply files and config_nerf.yml in {root_dir}")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ply") or file == "config_nerf.yml":
                file_path = os.path.join(root, file)
                os.remove(file_path)

def remove_unecessary_files(root_dir):
    logging.info(f"Removing unnecessary files in {root_dir}")
    remove_jpg_dirs(root_dir)
    remove_ply_and_config(root_dir)

def create_required_out_folders(out_folder):
    # Delete the directory if it exists
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)

    pose_overlayed_rgb_dir = f"{out_folder}/pose_overlayed_rgb"
    ob_in_cam_dir = f"{out_folder}/ob_in_cam"

    for _dir in [out_folder, pose_overlayed_rgb_dir, ob_in_cam_dir]:
        os.makedirs(_dir, exist_ok=True)
        os.chmod(_dir, 0o777)
    
    return out_folder, pose_overlayed_rgb_dir, ob_in_cam_dir


def process_frames(args, out_folder, use_segmenter=False, segmenter=None, cfg_bundletrack=None):
    """
    Processes frames from a reader and runs tracking and pose estimation.

    Args:
        args: Arguments object with stride parameter.
        out_folder: Path to save output pose files.
        use_segmenter: Whether to use a segmenter for masks.
        segmenter: Segmenter object (if used).
        cfg_bundletrack: Configuration dict for BundleTrack (optional).
    """

    out_folder, pose_overlayed_rgb_dir, ob_in_cam_dir = create_required_out_folders(out_folder)
    cfg_bundletrack = yaml.load(open(f"{code_dir}/BundleTrack/config_ho3d.yml", "r"))
    cfg_bundletrack["SPDLOG"] = int(args.debug_level)
    # cfg_bundletrack['depth_processing']["zfar"] = 1
    cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["erode_mask"] = 3
    cfg_bundletrack["debug_dir"] = out_folder + "/"
    cfg_bundletrack["bundle"]["max_BA_frames"] = 10
    cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
    cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
    cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
    cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
    cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
    cfg_bundletrack["feature_corres"]["map_points"] = True
    cfg_bundletrack["feature_corres"]["resize"] = 400
    cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = True
    cfg_bundletrack["keyframe"]["min_rot"] = 5
    cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
    cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
    cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
    cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
    cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
    cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
    cfg_bundletrack["p2p"]["max_dist"] = 0.02
    cfg_bundletrack["p2p"]["max_normal_angle"] = 45
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = yaml.load(open(f"{code_dir}/config.yml", "r"))
    cfg_nerf["continual"] = True
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["mesh_resolution"] = 0.005
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
    cfg_nerf["notes"] = ""
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, "w"))

    # if use_segmenter:
    #   segmenter = Segmenter()

    tracker = BundleSdf(
        cfg_track_dir=cfg_track_dir,
        cfg_nerf_dir=cfg_nerf_dir,
        start_nerf_keyframes=5,
        use_gui=use_gui,
    )

    reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)

    K = reader.K.copy()
    print(f"camera intrinsics: {K}")

    _pose_in_model = np.eye(4)

    for i in range(0, len(reader.color_files), args.stride):
        color_file = reader.color_files[i]
        color = cv2.imread(color_file)
        H0, W0 = color.shape[:2]
        depth = reader.get_depth(i)
        H, W = depth.shape[:2]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

        # Mask selection
        if i == 0:
            mask = reader.get_mask(0)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            if use_segmenter and segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
        else:
            if use_segmenter and segmenter:
                mask = segmenter.run(color_file.replace("rgb", "masks"))
            else:
                mask = reader.get_mask(i)
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # Erode mask if configured
        if cfg_bundletrack and cfg_bundletrack.get("erode_mask", 0) > 0:
            kernel_size = cfg_bundletrack["erode_mask"]
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)

        id_str = reader.id_strs[i]
        _filename, _pose_in_model, _frames = tracker.run(
            color,
            depth,
            K,
            id_str,
            mask=mask,
            occ_mask=None,
            pose_in_model=_pose_in_model,
        )

        print(_pose_in_model)
        filename = id_str.split('.')[0]
        out_pose_path = os.path.join(out_folder, "ob_in_cam", f"{filename}.txt")
        os.makedirs(os.path.dirname(out_pose_path), exist_ok=True)
        np.savetxt(out_pose_path, _pose_in_model, fmt="%.6f")

        # Save overlayed RGB
        row1_rgb = np.array(_frames["row1"]["rgb"])[..., :3]

        PILImage.fromarray(row1_rgb).save(
            os.path.join(pose_overlayed_rgb_dir, f"{filename}.png")
        )

    remove_unecessary_files(out_folder)
    tracker.on_finish()

    return out_folder, pose_overlayed_rgb_dir, ob_in_cam_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="../realworld")
    parser.add_argument("--use_segmenter", type=int, default=0)
    parser.add_argument("--use_gui", type=int, default=0)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="interval of frames to run; 1 means using every frame",
    )
    parser.add_argument(
        "--debug_level", type=int, default=0, help="higher means more logging"
    )
    args = parser.parse_args()

    video_dir = args.video_dir
    use_segmenter = args.use_segmenter
    use_gui = args.use_gui

    set_seed(0)

    obj_prompt_mapper_data = {}

    for obj_idx, src in enumerate(glob.glob(f"{video_dir}/out/samv2/*/obj_masks")):
        dst = f"{video_dir}/masks"

        # If dst exists, remove it
        if os.path.exists(dst):
            if os.path.islink(dst):
                os.unlink(dst)
            elif os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)

        # Create new symlink
        logging.info(f"Creating symlink from {src} to {dst}")
        os.symlink(src, dst)
        print(f"Symlink created from {src} to {dst}")      

        _out_folder, pose_overlayed_rgb_dir, ob_in_cam_dir = process_frames(
            args=args,
            out_folder=f"{video_dir}/out/bundlesdf/demonstration/obj_{obj_idx + 1}",
            use_segmenter=False,
            segmenter=None,
            cfg_bundletrack={"erode_mask": 3}  # Optional
        )

        # save obj_prompt_mapper.json
        src_list=src.split('/')
        obj_prompt_mapper_data[f"obj_{obj_idx + 1}"] = src_list[src_list.index('samv2')+1]
    
    with open(f"{os.path.dirname(_out_folder)}/obj_prompt_mapper.json", "w") as f:
        json.dump(obj_prompt_mapper_data, f, indent=2)

# python run_pose_only_bsdf.py --video_dir /home/jishnu/task_26_crackerbox.new --out_folder /home/jishnu/task_26_crackerbox.new/demo_obj_poses