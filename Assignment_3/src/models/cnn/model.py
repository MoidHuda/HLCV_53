import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super().__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        layers.append(nn.Conv2d(self.input_size, self.hidden_layers[0], kernel_size=3, stride=1, padding=1))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(self.activation())

        for i in range(1, len(self.hidden_layers) - 1):
            layers.append(nn.Conv2d(self.hidden_layers[i-1], self.hidden_layers[i], kernel_size=3, stride=1, padding=1))
            if self.norm_layer is not None:
                layers.append(self.norm_layer(self.hidden_layers[i]))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(self.activation())
            if self.drop_prob > 0:
                layers.append(nn.Dropout(self.drop_prob))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))
        self.model = nn.Sequential(*layers)

        # in_channels = self.input_size
        # size = 32
        # for out_channels in self.hidden_layers:
        #     layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        #     layers.append(self.activation())
        #     layers.append(nn.MaxPool2d(kernel_size=2, stride=2,padding=1))
        #     in_channels = out_channels
        #     size = ((size + 2 * 1 - 2) // 2) + 1

        # layers.append(nn.Flatten())
        # layers.append (nn.Linear(self.hidden_layers[-1] * size * size, self.num_classes))
        # self.layers = nn.Sequential(*layers)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # filters = self.model[0].weight.data.cpu().numpy()
        # filters = self._normalize(filters)
        # num_filters = filters.shape[0]
        # filter_size = filters.shape[2]
        # grid_size = int(np.ceil(np.sqrt(num_filters)))
        # print(f"Number of filters: {num_filters}, Filter size: {filter_size}, Grid size: {grid_size}")
        # grid = np.zeros((grid_size * (filter_size + 1), grid_size * (filter_size + 1), 3), dtype=np.float32)
        # for i in range(num_filters):
        #     row = i // grid_size
        #     col = i % grid_size
        #     start_row = row * (filter_size + 1)
        #     start_col = col * (filter_size + 1)
        #     grid[start_row:start_row + filter_size, start_col:start_col + filter_size, :] = filters[i].transpose(1, 2, 0)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(grid)
        # plt.axis('off')
        
        fig, axis = plt.subplots(8, 16, figsize=(16, 8))
        filters = self.model[0].weight.data.cpu().numpy().shape[0]
        for i in range(8 * 16):
            plot = axis[i // 16, i % 16]
            if i < filters:
                filter_img = self.model[0].weight.data.cpu().numpy()[i]
                normalized_img = self._normalize(filter_img)
                plot.imshow(normalized_img.transpose(1, 2, 0))
            plot.axis('off')
        plt.show()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        out = self.model(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
