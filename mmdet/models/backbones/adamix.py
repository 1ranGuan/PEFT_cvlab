# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn


class Adamix(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 num_of_adapters=4):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.num_of_adapters = num_of_adapters
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_projs = nn.ModuleList(nn.Linear(self.n_embd, self.down_size) for _ in range(num_of_adapters))
        self.non_linear_func = nn.ReLU()
        self.up_projs = nn.ModuleList(nn.Linear(self.down_size, self.n_embd) for _ in range(num_of_adapters))

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                for i in range(num_of_adapters):
                    nn.init.kaiming_uniform_(self.down_projs[i].weight, a=math.sqrt(5))
                    nn.init.zeros_(self.up_projs[i].weight)
                    nn.init.zeros_(self.down_projs[i].bias)
                    nn.init.zeros_(self.up_projs[i].bias)
    
    def _gengrate_expert_ids(self):
        expert_ids = torch.randint(0, self.num_of_adapters, (2,))
        return expert_ids

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down_idx, up_idx = self._gengrate_expert_ids()
        down = self.down_projs[down_idx](x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_projs[up_idx](down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output