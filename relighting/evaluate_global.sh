# relighting of GT
blender --background ./model/bunny_test_global.blend -P bunny_render_global.py -- --method GT --dataroot ../data/synthetic/global_lighting --result_dir ./results/global/gt
# relighting of our pipeline
blender --background ./model/bunny_test_global.blend -P bunny_render_global.py -- --method SOLD --dataroot ../estimating/results/global --result_dir ./results/global/sold
# calculate relighting errors
python ./bunny_qualitative_global.py --dataroot ./results/global/gt --result_dir ./results/global/sold