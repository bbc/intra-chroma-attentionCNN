scheme = 3
deploy_path = "/work/marcb/Intra_Chroma/deploy"
config = "config/norelu.py"

bit_depth = 10

scale_luma = 20
scale_bound1 = 21
scale_bound2 = 26
scale_attb = 25
scale_attx = 21
scale_attx1 = 23
scale_head_in = 16  # computed from scales and shifts of attention module
scale_head_out = 21

shift_luma = 9
shift_bound1 = 6
shift_bound2 = 10
