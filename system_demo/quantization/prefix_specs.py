"""Observed fixed-shape boundaries of the exported ViT-L/14 diagnostic graph."""
ATTN='/transformer/resblocks/resblocks.0/attn/'
SPECS={
    'stem':('/ln_pre/LayerNormalization_output_0',[1,257,1024]),
    'block0':('/transformer/resblocks/resblocks.0/Add_1_output_0',[257,1,1024]),
    'ln1':('/transformer/resblocks/resblocks.0/ln_1/LayerNormalization_output_0',[257,1,1024]),
    'attn0':('/transformer/resblocks/resblocks.0/Add_output_0',[257,1,1024]),
    'qkv':(ATTN+'Add_output_0',[257,1,3072]),
    'q':(ATTN+'Reshape_6_output_0',[1,16,257,64]),
    'k':(ATTN+'Reshape_7_output_0',[1,16,257,64]),
    'v':(ATTN+'Reshape_8_output_0',[1,16,257,64]),
    'core':(ATTN+'MatMul_2_output_0',[1,16,257,64]),
    'projection':(ATTN+'Reshape_10_output_0',[257,1,1024]),
}

def directory_name(segment,cpu_attention=False,optimization_level=3):
    value='prefix_control_20261010' if segment=='stem' else 'prefix_'+segment+'_control_20261010'
    if cpu_attention: value=value.replace('_control_','_sdpa_cpu_control_')
    if optimization_level==0: value=value.replace('_control_','_opt0_control_')
    return value
