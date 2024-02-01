import numpy as np
from local_attention import LocalAttention, LocalTransformer

from mindspore import Tensor, ops

# # Ex 1
# q = ops.randn((2, 8, 2048, 64))
# k = ops.randn((2, 8, 2048, 64))
# v = ops.randn((2, 8, 2048, 64))
#
# # save input
# np.save("D://桌面//q.npy", q.asnumpy())
# np.save("D://桌面//k.npy", k.asnumpy())
# np.save("D://桌面//v.npy", v.asnumpy())
#
# attn = LocalAttention(
#     dim=64,  # dimension of each head (you need to pass this in for relative positional encoding)
#     window_size=512,  # window size. 512 is optimal, but 256 or 128 yields good enough results
#     causal=True,  # auto-regressive or not
#     look_backward=1,  # each window looks at the window before
#     look_forward=0,  # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
#     dropout=0,  # post-attention dropout
#     exact_windowsize=False,  # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
# )
#
# mask = ops.ones((2, 2048)).bool()
# out = attn(q, k, v, mask=mask)  # (1, 8192, 256)
# np.save("D://桌面//out.npy", out.asnumpy())
# # get relative diff: rel_diff=abs((out.detach().numpy()-np.load("D://桌面//out.npy"))/torch.where(out!=0, out, out.mean()).detach().numpy()).mean()
# # 3.4579991e-06
#
#
# # Ex 2
# qk = ops.randn((2, 8, 2048, 64))
# v = ops.randn((2, 8, 2048, 64))
#
# # save input
# np.save("D://桌面//qk.npy", qk.asnumpy())
# np.save("D://桌面//v.npy", v.asnumpy())
#
# attn = LocalAttention(dim=64, window_size=512, shared_qk=True, causal=True)
#
# mask = ops.ones((2, 2048)).bool()
# out = attn(qk, qk, v, mask=mask)  # (2, 8, 2048, 64)
# np.save("D://桌面//out.npy", out.asnumpy())
# # rel_diff: 2.9060013e-06
#
#
# # Ex 3
# q = ops.randn((8, 2057, 64))
# k = ops.randn((8, 2057, 64))
# v = ops.randn((8, 2057, 64))
#
# # save input and output
# np.save("D://桌面//q.npy", q.asnumpy())
# np.save("D://桌面//k.npy", k.asnumpy())
# np.save("D://桌面//v.npy", v.asnumpy())
#
# attn = LocalAttention(
#     window_size=512, causal=True, autopad=True  # auto pads both inputs and mask, then truncates output appropriately
# )
#
# mask = ops.ones((1, 2057)).bool()
# out = attn(q, k, v, mask=mask)  # (8, 2057, 64)
# np.save("D://桌面//out.npy", out.asnumpy())
# # rel_diff: 5.1276957e-06

# Ex 4
model = LocalTransformer(num_tokens=256, dim=512, depth=6, max_seq_len=8192, causal=True, local_attn_window_size=256)

x = ops.randint(0, 256, (1, 8192))

params = model.parameters_dict()
pt_params = {}
n = 0
for name in params:
    n = n + 1
    p = params[name]
    if name.endswith(".beta"):
        name = name[: name.rfind(".beta")] + ".bias"
    if name.endswith(".gamma"):
        name = name[: name.rfind(".gamma")] + ".weight"
    if name.endswith(".moving_mean"):
        name = name[: name.rfind(".moving_mean")] + ".running_mean"
    if name.endswith(".moving_variance"):
        name = name[: name.rfind(".moving_variance")] + ".running_var"
    if name.endswith(".embedding_table"):
        name = name[: name.rfind(".embedding_table")] + ".weight"
    if name[0].isdigit():
        name = "layers." + name
    if n >= 11 and "layers.0" in name:
        name = name.replace("layers.0", "layers")
    pt_params[name] = p.value().asnumpy()
np.save("D://桌面//params.npy", pt_params)
np.save("D://桌面//x.npy", x.asnumpy())

logits = model(x)
np.save("D://桌面//logits.npy", logits.asnumpy())
# rel_diff: 6.2234226e-06
