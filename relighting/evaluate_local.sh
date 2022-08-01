# relighting of GT
blender --background ./model/bunny_test_local.blend -P bunny_render_local.py -- --method GT --dataroot ../data/synthetic/local_pano --result_dir ./results/local/gt
# relighting of our pipeline
blender --background ./model/bunny_test_local.blend -P bunny_render_local.py -- --method SOLD --dataroot ../estimating/results/local --result_dir ./results/local/sold
# calculate relighting errors
python ./bunny_qualitative_local.py --dataroot ./results/local/gt --result_dir ./results/local/sold
