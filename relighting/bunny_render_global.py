import bpy
import os
import sys
import glob
import shutil
import argparse
import imageio
import numpy as np
from tqdm import tqdm

def boolean_string(s):
    if s == '1':
        return True
    if s == '0':
        return False
    if s.lower() == 'true':
        return True
    return False

def check_aspect_ratio(env_path, tmp_path, method='GT'): # only use upper part in global lighting and set lower half all black
    if method == 'GT':
        env_data = imageio.imread(env_path)*4.0 # NOTE: fix scale for global GT
    else:
        env_data = imageio.imread(env_path)
    H, W, C = env_data.shape
    if H<W//2:
        new_env_data = np.zeros((W//2, W, C), dtype=np.float32)
        new_env_data[:H,:,:] = env_data
        env_filename = os.path.split(env_path)[-1]
        new_env_save_path = os.path.join(tmp_path, env_filename)
        imageio.imsave(new_env_save_path, new_env_data)
        return new_env_save_path
    else: 
        new_env_data = env_data
        new_env_data[H//2:,:,:] = 0
        env_filename = os.path.split(env_path)[-1]
        new_env_save_path = os.path.join(tmp_path, env_filename)
        imageio.imsave(new_env_save_path, new_env_data)
        return new_env_save_path

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="../data/synthetic/global_lighting")
parser.add_argument("--splitfile", type=str, default='../data/synthetic/split/test.txt')
parser.add_argument("--result_dir", type=str, default='./results/global/gt')
parser.add_argument("--method", type=str, choices=['GT', 'SOLD'])
parser.add_argument("--persistent_data", type=bool, default=True)
parser.add_argument("--release_persistent_every", type=int, default=500)
parser.add_argument("--check_aspect_ratio", default=True)

# print(sys.argv)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

# env_files = get_env_maps(args.result_dir, args.name, args.epoch)
with open(args.splitfile, 'r') as f:
    items = [line.strip().split() for line in f.readlines() if line.strip()]
render_save_dir = os.path.join(args.result_dir, 'relight_bunny')
render_tmp_dir = os.path.join(args.result_dir, '.render_tmp')
os.makedirs(render_save_dir, exist_ok=True)
os.makedirs(render_tmp_dir, exist_ok=True)

# Set GPU
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
# add GPU devices to render
preferences = bpy.context.preferences
cycles_preferences = preferences.addons['cycles'].preferences
cuda_devices, opencl_devices = cycles_preferences.get_devices()
cycles_preferences.compute_device_type = "CUDA"
bpy.context.scene.render.use_persistent_data = args.persistent_data # NOTE: memory for time, much faster!
# Set tmp dir
bpy.data.scenes["Scene"].node_tree.nodes["Env Output"].base_path = render_tmp_dir

for i, item in enumerate(tqdm(items)):
    city_cam_name, sky_name, angle_id = item
    filebasename = '%06d_%s_%s_%s'%(i+1, city_cam_name, sky_name, angle_id)

    if args.method == 'GT':
        global_env_path = os.path.join(args.dataroot, city_cam_name, 'rendered', sky_name, 'global_lighting_%s.hdr' %(angle_id))
    elif args.method == 'SOLD':
        global_env_path = os.path.join(args.dataroot, 'dump', '%s_global_est.hdr' %(filebasename))

    # NOTE: change environment map here
    if i%args.release_persistent_every == 0: # release persistent data regularly to avoid memory leak
        bpy.context.scene.render.use_persistent_data = False # NOTE: release unused persistent envmap data, otherwise may cause memory leak!
    bpy.data.images.remove(bpy.context.scene.world.node_tree.nodes['equi_main'].image) # delete old environment map
    if args.check_aspect_ratio:
        new_env_path = check_aspect_ratio(global_env_path, render_tmp_dir, args.method)
        bpy.context.scene.world.node_tree.nodes['equi_main'].image = bpy.data.images.load(new_env_path) # load new environment map
    else:
        bpy.context.scene.world.node_tree.nodes['equi_main'].image = bpy.data.images.load(global_env_path) # load new environment map
    if i%args.release_persistent_every == 0:
        bpy.context.scene.render.use_persistent_data = args.persistent_data # NOTE: memory for time, much faster!

    save_path = os.path.join(render_save_dir, '%s_bunny_relight.hdr' %(filebasename))
    bpy.context.scene.render.filepath = save_path # set render output path
    bpy.ops.render.render(write_still=True)  # this line will automatically save .png ldr image to save_path (change extension name)
    bunny_src = glob.glob(os.path.join(render_tmp_dir, 'Image*'))[0]
    shutil.move(bunny_src, save_path)
    if args.check_aspect_ratio and new_env_path.find(render_tmp_dir) != -1:
        os.remove(new_env_path)
