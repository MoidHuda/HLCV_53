import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod # To be implemented by child classes.
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()
    
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ret_str += f"\nNumber of trainable parameters: {num_params:,d}"
        #################################################
        sequential = list(self.children())[0]
        children = list(sequential.children())
        if len(children) == 0:
            ret_str += "  - No layers found.\n"
            return ret_str
        ret_str += "\n" + "=" * 80 + "\n"
        ret_str += "Model Summary:\n"
        ret_str += f"{self.__class__.__name__} with {len(children)} layers:\n"
        for name, module in sequential.named_children():
            ret_str += f"  - {name}: {module.__class__.__name__}, {sum([i.numel() for i in module.parameters()])}\n"
        ret_str += "=" * 80 + "\n"
        ret_str += f"Total number of parameters: {sum(p.numel() for p in self.parameters()):,d}\n"
        ret_str += f"Total number of trainable parameters: {num_params:,d}\n"
        ret_str += "=" * 80 + "\n"
        return ret_str