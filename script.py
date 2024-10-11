import os

# GPT 3x_to_x 4层
gpu_id = 3
seed = 1
target = '3x_to_x'
model='GPT'
scheduler = None
train_batch_size = 256
test_batch_size = 256
N_train = 2048
N_test = 256
d_model = 32
d_ff = 32
n_layer = 4

lr = 2e-4
n_epoch = 500
d_head = 4
n_heads = 128
init_var_ff = 1e-3
init_var_attn = 1e-3
plot_headoutput_similarity_epoch = 20
save_model_epoch = 20
threshold = 0.9

dir_suffix = 'test_L4'
suffix = 'nh_{}-dh_{}-varff_{}-varattn_{}-lr_{}'.format(n_heads, d_head, init_var_ff, init_var_attn, lr)
os.system(f'CUDA_VISIBLE_DEVICES={gpu_id} /bin/python -m main -N_train {N_train} -N_test {N_test} -train_bs {train_batch_size} -test_bs {test_batch_size} -sme {save_model_epoch}\
          -seed {seed} -m {model} -func {target} -lr {lr} -scheduler {scheduler} -ne {n_epoch} -nl {n_layer} -nh {n_heads} -dh {d_head} -dm {d_model} -phst {threshold} \
          -d_ff {d_ff} -init_var_ff {init_var_ff} -init_var_attn {init_var_attn} -phse {plot_headoutput_similarity_epoch} -dir_suffix {dir_suffix} -suffix {suffix}')


