from torch._C import LiteScriptModule
from cnnexplorer.utils import ActivationSampler
import torch
import matplotlib.pyplot as plt


class ExtractCnnMaps:
    """This class provides methods to extract feature maps and kernels from a Convolutional Neural Network model.
    to display them.

     Parameters:
    -----------
    model : torch.nn.Module
        The PyTorch model from which layer information will be extracted.
    dataset :, optional
        The dataset from which the images will be extracted. Default is None.
    image : torch.Tensor, optional
        The image to be used to extract the feature maps. Default is None.
    device : str, optional
        The device to be used to extract the feature maps. Default is 'cpu'.


    Methods
    -----------
    get_feature_maps()
    get_multiple_feature_maps()
    get_kernels()
    get_multiple_kernels()
    show_feature_maps()
    show_kernels_per_channel()
    show_channel_kernels()

    """

    # TODO remover a entrada do dataset. A imagem já basta.

    def __init__(
            self,
            model: torch.nn.Module,
            dataset=None,
            image: torch.Tensor = None,
            device: str = 'cpu'
    ):
        assert (dataset is None and image is not None) or (
                dataset is not None and image is None), f'dataset and image cannot both be None or have content. Only use ' \
                                                        f'dataset or image.'
        if image is not None:
            assert type(image) is torch.Tensor, f'Image is not a torch.Tensor type'

        self.model = model
        self.dataset = dataset
        self.image = image
        self.device = device

    def get_feature_maps(
            self,
            layer: torch.nn.Module,
            img_idx: int = None
    ):
        """Extracts the feature maps from a given layer. To extract the feature maps, the model must be run on an image and
        the layer must be activated. The image can be provided as an input or the dataset can be used. If the dataset is used,
        the image index must be provided.

        The used method is based on the Pytorch Hooks. The ActivationSampler class is used to extract the feature maps.

        Parameters:
        -----------
        layer : torch.nn.Module
            The layer from which the feature maps will be extracted.
        img_idx : int, optional
            The index of the image to be used to extract the feature maps. Default is None.

        Returns:
        --------
        layer_feature_maps : torch.Tensor
            The feature maps extracted from the layer.
        """

        sampler = ActivationSampler(layer)

        if self.image is not None:
            with torch.no_grad():
                img = self.image.to(self.device)[None]
                self.model(img)
        else:
            img, label = self.dataset[img_idx]
            with torch.no_grad():
                img = img.to(self.device)[None]
                self.model(img)

        layer_feature_maps = sampler().to(self.device)[0]

        return layer_feature_maps

        # TODO: Descrever a função

    def get_multiple_feature_maps(
            self,
            layers: list,
            img_idx: int = None
    ):
        """ Call the get_feature_maps function for each layer in the list of layers and return a list of feature maps. Each
        index of the list corresponds to the feature maps extracted from the layer in the same index.

        Parameters:
        -----------
        layers : list of torch.nn.Module
            The list of layers from which the feature maps will be extracted.
        img_idx : int, optional
            The index of the image to be used to extract the feature maps. Default is None.

        Returns:
        --------
        layers_fm_list : list of torch.Tensor
            The list of feature maps extracted from the layers.
        """

        layers_fm_list = []

        if self.image is not None:
            for i in range(len(layers)):
                layers_fm_list.append(self.get_feature_maps(layers[i], None))
        else:
            for i in range(len(layers)):
                layers_fm_list.append(self.get_feature_maps(layers[i], img_idx))

        return layers_fm_list

    def get_kernels(
            self,
            layer: torch.nn.Module
    ):
        """Extracts the kernels from a given layer. The layer must be a Convolutional layer.
        To extract the kernels, the layer weights are extracted using the layer.weight attribute.

        Parameters:
        -----------
        layer : torch.nn.Module
            The layer from which the kernels will be extracted.

        Returns:
        --------
        kernels_to_cpu : torch.Tensor
            The kernels extracted from the layer.
        """
        kernels = layer.weight
        kernels_to_cpu = kernels.detach().to('cpu')
        return kernels_to_cpu

    def get_multiple_kernels(self, layers: list):
        """ Call the get_kernels function for each layer in the list of layers and return a list of kernels.
        Each index of the list corresponds to the kernels extracted from the layer in the same index.

        Parameters:
        -----------
        layers : list of torch.nn.Module
            The list of layers from which the kernels will be extracted.

        Returns:
        --------
        kernels_list : list of torch.Tensor
            The list of kernels extracted from the layers.
        """

        kernels_list = []
        for i in range(len(layers)):
            kernels_list.append(self.get_kernels(layers[i]))

        return kernels_list

    # FIX: When img_idx = None, for some reason the map 56 is not being printed.
    # TODO: Implement the possbility to save the figure in a given extension in a given directory.
    def show_feature_maps(
            self,
            layers: list,
            layers_fm_list: list,
            img_idx: int = None,
            maps_idx: list = None,
            scalar_data: bool = False,
            fig_size: tuple = (20, 75),
            ncols: int = 4,
            n_first_maps: int = 64,
            plot_title: bool = True
    ) -> None:
        """Show the feature maps extracted from a list of given layers. The feature maps can be provided as an input or the
        get_feature_maps function can be used to extract the feature maps.

        Parameters:
        -----------
        layers : list of torch.nn.Module
            The list of layers from which the feature maps will be extracted. Default is None.
        layers_fm_list : list of torch.Tensor
            The list of feature maps extracted from the layers. Default is None.
        img_idx : int, optional
            The index to identify the image from a dataset. Default is None.
        maps_idx : list of int, optional
            The list of feature maps indexes to be shown. Default is None.
        scalar_data : bool, optional
            If True, it's used to get the min and max values from the feature maps, so the plots can be normalized.
        fig_size : tuple of int, optional
            The figure size. Default is (20, 75).
        ncols : int, optional
            The number of columns in the figure. Default is 4.
        n_first_maps : int, optional
            The number of the first feature maps to be shown, case maps_idx is None. Default is 64.
        plot_title : bool, optional
            If True, the title of the figure will be shown. Default is True.
        """

        n_layers = len(layers_fm_list)
        nrows = n_first_maps // ncols

        if maps_idx is not None:

            qty_maps = len(maps_idx)

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data:
                    # Get min and max from set of feature maps
                    v_min = layers_fm_list[layer_idx].min()
                    v_max = layers_fm_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(qty_maps):
                    # Show feature map
                    map_idx = maps_idx[idx]
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(layers_fm_list[layer_idx][map_idx], vmin=v_min, vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    if plot_title:
                        plt.title(f'Image {img_idx} \nFeature map {map_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

        else:

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data:
                    # Get min and max from set of feature maps
                    v_min = layers_fm_list[layer_idx].min()
                    v_max = layers_fm_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(n_first_maps):
                    # Show feature map
                    map_idx = idx
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(layers_fm_list[layer_idx][map_idx], vmin=v_min, vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    if plot_title:
                        plt.title(f'Image {img_idx} \nFeature map {map_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

    # TODO: Implement the possbility to save the figure in a given extension in a given directory.
    def show_kernels_per_channel(
            self,
            layers: list,
            kernels_list: list,
            kernels_idx: list = None,
            channel_idx: int = 0,
            scalar_data: bool = False,
            fig_size: tuple = (20, 75),
            ncols: int = 4,
            n_first_kernels: int = 64,
            plot_title: bool = True
    ) -> None:

        """Show the kernels extracted from a list of given layers for a specified channel. The kernels can be provided as
        an input or the get_kernels function can be used to extract the kernels.
        Parameters:
        -----------
        layers : list of torch.nn.Module
            The list of layers from which the kernels will be extracted.
        kernels_list : list of torch.Tensor
            The list of kernels extracted from the layers.
        kernels_idx : list of int, optional
            The list of kernels indexes to be shown. Default is None.
        channel_idx : int, optional
            The index of the channel to be shown. Default is 0.
        scalar_data : bool, optional
            If True, it's used to get the min and max values from the kernels, so the plots can be normalized.
        fig_size : tuple of int, optional
            The figure size. Default is (20, 75).
        ncols : int, optional
            The number of columns in the figure. Default is 4.
        n_first_kernels : int, optional
            The number of the first kernels to be shown, case kernels_idx is None. Default is 64.
        plot_title : bool, optional
            If True, the title of the figure will be shown. Default is True.

        """

        n_layers = len(kernels_list)
        nrows = n_first_kernels // ncols

        if kernels_idx is not None:

            qty_maps = len(kernels_idx)

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(qty_maps):
                    # Show kernel
                    kernel_idx = kernels_idx[idx]
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmin=v_min, vmax=v_max,
                                    cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    if plot_title:
                        plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

        else:

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(n_first_kernels):
                    # Show kernel
                    kernel_idx = idx
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmin=v_min, vmax=v_max,
                                    cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    if plot_title:
                        plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

    def show_channels_per_kernel(
            self,
            layers: list,
            kernels_list: list,
            kernel_idx: int = 0,
            channels_idx: list = None,
            scalar_data: bool = False,
            fig_size: tuple = (20, 75),
            ncols: int = 4,
            n_first_kernels: int = 64,
            plot_title: bool = True
    ) -> None:
        """Show the channels extracted from a list of given layers for a specified kernel. The kernels can be provided as
        an input or the get_kernels function can be used to extract the kernels.
        Parameters:
        -----------
        layers : list of torch.nn.Module
            The list of layers from which the kernels will be extracted.
        kernels_list : list of torch.Tensor
            The list of kernels extracted from the layers.
        kernel_idx : int, optional
            The index of the kernel to be shown. Default is 0.
        channels_idx : list of int, optional
            The list of channels indexes to be shown. Default is None.
        scalar_data : bool, optional
            If True, it's used to get the min and max values from the kernels, so the plots can be normalized.
        fig_size : tuple of int, optional
            The figure size. Default is (20, 75).
        ncols : int, optional
            The number of columns in the figure. Default is 4.
        n_first_kernels : int, optional
            The number of the first kernels to be shown, case kernels_idx is None. Default is 64.
        plot_title : bool, optional
            If True, the title of the figure will be shown. Default is True.

        """

        n_layers = len(kernels_list)
        nrows = n_first_kernels // ncols

        if channels_idx is not None:

            qty_channel = len(channels_idx)

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(qty_channel):
                    # Show kernel
                    channel_idx = channels_idx[idx]
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    if plot_title:
                        plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()

        else:

            for layer_idx in range(n_layers):

                plt.figure(figsize=fig_size)

                if scalar_data:
                    # Get min and max from set of kernels
                    v_min = kernels_list[layer_idx].min()
                    v_max = kernels_list[layer_idx].max()
                else:
                    v_min = None
                    v_max = None

                for idx in range(n_first_kernels):
                    # Show kernel
                    channel_idx = idx
                    fig = plt.subplot(nrows, ncols, idx + 1)
                    ax = plt.imshow(kernels_list[layer_idx][kernel_idx][channel_idx], vmax=v_max, cmap='gray')
                    layer_path = layers['layer_path'][layer_idx]
                    if plot_title:
                        plt.title(f'Kernel {kernel_idx} - Channel {channel_idx} - {layer_path}')
                    # Hide axis
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    # Adjust space between plots
                    plt.subplots_adjust(wspace=0.02, hspace=0.0)
                    plt.tight_layout()
