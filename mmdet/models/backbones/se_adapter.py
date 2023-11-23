# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.init as init

class SEAdapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # Squeeze部分，全局平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation部分，全连接层
        self.excitation = nn.Sequential(
            nn.Linear(self.n_embd, self.down_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.down_size , self.n_embd),
            nn.Sigmoid()
        )

        self.dropout = dropout
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
            # 使用Xavier/Glorot初始化
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                        init.zeros_(m.bias)


    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        #把x的最后一维提前
        res = x
        x = x.permute(0, 3, 1, 2)
        squeeze_output = self.squeeze(x)
        squeeze_output = squeeze_output.view(x.size(0), -1)
        
        # Excitation
        excitation_output = self.excitation(squeeze_output)
        excitation_output = excitation_output.view(x.size(0), x.size(1), 1, 1)
        
        # Scale
        up = x * excitation_output
        up = up.permute(0, 2, 3, 1)
        res = self.up_proj(self.non_linear_func(self.down_proj(res)))
        up = up+res
        up = up*self.scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output