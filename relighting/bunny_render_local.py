import bpy
import os
import sys
import glob
import shutil
import argparse
import imageio
from tqdm import tqdm


def boolean_string(s):
    if s == '1':
        return True
    if s == '0':
        return False
    if s.lower() == 'true':
        return True
    return False

def check_local_pano(env_path, tmp_path):
    env_data = imageio.imread(env_path)*4.0 # NOTE: fix scale for local GT
    env_data[env_data<0.0] = 0.0
    env_data[env_data>2000.0] = 2000.0
    env_filename = os.path.split(env_path)[-1]
    new_env_save_path = os.path.join(tmp_path, env_filename)
    imageio.imsave(new_env_save_path, env_data)
    return new_env_save_path

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="../data/synthetic/local_pano/")
parser.add_argument("--splitfile", type=str, default='../data/synthetic/split/test.txt')
parser.add_argument("--result_dir", type=str, default='./results/local/gt')
parser.add_argument("--method", type=str, choices=['GT', 'SOLD'])
parser.add_argument("--persistent_data", type=bool, default=True)
parser.add_argument("--release_persistent_every", type=int, default=500)

args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

with open(args.splitfile, 'r') as f:
    _items = [line.strip().split() for line in f.readlines() if line.strip()]
items = []
for _item in _items:
    city_cam_name, sky_name, angle_id = _item
    for i in range(4):
        items.append([city_cam_name, sky_name, angle_id, str(i)])

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
    city_cam_name, sky_name, angle_id, local_id = item
    filebasename = '%06d_%s_%s_%s'%(i//4+1, city_cam_name, sky_name, angle_id)
    if args.method == 'GT':
        local_env_path = os.path.join(args.dataroot, city_cam_name, 'rendered', sky_name, 'angle_%s_camera_%s.hdr' %(angle_id, local_id))
    elif args.method == 'SOLD':
        local_env_path = os.path.join(args.dataroot, 'dump', '%s_local_%s_est.hdr' %(filebasename, local_id))
    if args.method == 'GT':
        tmp_local_env_path = check_local_pano(local_env_path, render_tmp_dir)
    else:
        tmp_local_env_path = local_env_path

    # NOTE: change environment map here
    if i%args.release_persistent_every == 0: # release persistent data regularly to avoid memory leak
        bpy.context.scene.render.use_persistent_data = False # NOTE: release unused persistent envmap data, otherwise may cause memory leak!
    bpy.data.images.remove(bpy.context.scene.world.node_tree.nodes['equi_main'].image) # delete old environment map
    bpy.context.scene.world.node_tree.nodes['equi_main'].image = bpy.data.images.load(tmp_local_env_path) # load new environment map
    if i%args.release_persistent_every == 0:
        bpy.context.scene.render.use_persistent_data = args.persistent_data # NOTE: memory for time, much faster!

    save_path = os.path.join(render_save_dir, '%s_local_%s_bunny_relight.hdr' %(filebasename, local_id))
    bpy.context.scene.render.filepath = save_path # set render output path
    bpy.ops.render.render(write_still=True)  # this line will automatically save .png ldr image to save_path (change extension name)
    bunny_src = glob.glob(os.path.join(render_tmp_dir, 'Image*'))[0]
    shutil.move(bunny_src, save_path)

    if tmp_local_env_path.find(render_tmp_dir) != -1:
        os.remove(tmp_local_env_path)