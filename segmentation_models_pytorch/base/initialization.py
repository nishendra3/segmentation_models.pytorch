import torch.nn as nn
import math
INIT_A = 0.25

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            
            # alternate initialization from noemi
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
            nn.init.normal_(m.weight, mean=0, std=math.sqrt(2. / ((1 + math.pow(INIT_A, 2)) * n)))

            # nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        # if isinstance(m, (nn.Linear, nn.Conv2d)):
        #     nn.init.xavier_uniform_(m.weight)
        #     if m.bias is not None:
        #         nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        if isinstance(m, nn.Conv2d):
            
            # alternate initialization from noemi
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  
            nn.init.normal_(m.weight, mean=0, std=math.sqrt(2. / ((1 + math.pow(INIT_A, 2)) * n)))

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)