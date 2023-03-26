from typing import Tuple, Union, Any, Dict

from pandas import Series, DataFrame

from cnnexplorer.utils import feature_maps_interp, image_sampling, binary_dilation, feature_maps_masking, \
    get_image_label, remove_prefix
from cnnexplorer.layer_extractor import ExtractCnnLayers
from cnnexplorer.maps_extractor import ExtractCnnMaps

import pandas as pd
import numpy as np
import pyprog
import torch
import gc  # Garbage colector

# TODO: Improve documentation

def check_models_name(model1_name: str, model2_name: str):
    """Function to check if the names of the models are the same.
    If the names are the same, then the names are changed to model1_name(1) and model2_name(2).

    Parameters
    ----------
    model1_name : str
        Name of the first model.
    model2_name : str
        Name of the second model.

    Returns
    -------
    model1_name : str
        Name of the first model.
    model2_name : str
        Name of the second model.
    """
    if model1_name == model2_name:
        model1_name = model1_name + '(1)'
        model2_name = model2_name + '(2)'

    return model1_name, model2_name


def check_if_map_is_zero(
        feature_map1: torch.Tensor,
        feature_map2: torch.Tensor,
):
    """Function to check if the pair of feature maps are mostly composed of zero values. It's a treatment to avoid
  NaN correlations.

    Parameters
    ----------
    feature_map1 : torch.Tensor
        First feature map.
    feature_map2 : torch.Tensor
        Second feature map.

    Returns
    -------
    zero_flag_fm_1 : int
        Flag to indicate if the first feature map is mostly composed of zero values.
    zero_flag_fm_2 : int
        Flag to indicate if the second feature map is mostly composed of zero values.

  """
    # Calculate standard deviation of both feature maps
    std_fm_1 = feature_map1.std()
    std_fm_2 = feature_map2.std()

    if (std_fm_1 < 1e-10) and (std_fm_2 < 1e-10):
        # If std_fm < 1e-10, then zero_flag_fm = 1.
        zero_flag_fm_1 = 1
        zero_flag_fm_2 = 1
    elif (std_fm_1 < 1e-10) and (std_fm_2 >= 1e-10):
        zero_flag_fm_1 = 1
        zero_flag_fm_2 = 0
    elif (std_fm_1 >= 1e-10) and (std_fm_2 < 1e-10):
        zero_flag_fm_1 = 0
        zero_flag_fm_2 = 1
    else:
        zero_flag_fm_1 = 0
        zero_flag_fm_2 = 0

    return zero_flag_fm_1, zero_flag_fm_2


class CorrelationExplorer:
    """Class to explore the correlation between feature maps of two CNN models.

    Parameters
    ----------
    None

    Attributes
    ----------
    fm_corr_dict : dict
        Dictionary with the correlation between feature maps of two CNN models.
    min_fm_corr_dict : dict
        Dictionary with the minimum correlation between feature maps of two CNN models.
    max_fm_corr_dict : dict
        Dictionary with the maximum correlation between feature maps of two CNN models.
    stats_min_corr_dict : dict
        Dictionary with stats of the minimum correlation between feature maps of two CNN models.
    stats_max_corr_dict : dict
        Dictionary with stats the maximum correlation between feature maps of two CNN models.
    fm_corr : pandas.DataFrame
        Dataframe with the correlation between feature maps of two CNN models.
    min_fm_corr : pandas.DataFrame
        Dataframe with the minimum correlation between feature maps of two CNN models.
    max_fm_corr : pandas.DataFrame
        Dataframe with the maximum correlation between feature maps of two CNN models.
    stats_min_corr : pandas.DataFrame
        Dataframe with the stats minimum correlation between feature maps of two CNN models.
    stats_max_corr : pandas.DataFrame
        Dataframe with the stats maximum correlation between feature maps of two CNN models.

    Methods
    -------
    pearson_correlation()
    feature_maps_correlation()
    multiple_feature_maps_correlation()
    get_max_correlations()
    get_min_correlations()
    get_correlation_stats()
    correlation_pipeline()
    """

    def __init__(self):
        self.fm_corr_dict = None
        self.min_fm_corr_dict = None
        self.max_fm_corr_dict = None
        self.stats_min_corr_dict = None
        self.stats_max_corr_dict = None
        self.fm_corr = None
        self.min_fm_corr = None
        self.max_fm_corr = None
        self.stats_min_corr = None
        self.stats_max_corr = None

    def pearson_correlation(
            self,
            feature_map1: torch.Tensor,
            feature_map2: torch.Tensor,
    ):
        """Function to calculate Pearson correlation between two tensors.

        Parameters
        ----------
        feature_map1 : torch.Tensor
            First feature map.
        feature_map2 : torch.Tensor
            Second feature map.

        Returns
        -------
        corr : float
            Pearson correlation between two tensors.

      """
        # Concatenate two tensors along a new dimension.
        x = torch.stack([feature_map1, feature_map2])
        # Move tensor from CPU to GPU
        if torch.cuda.is_available():
            x = x.cuda()
        # Get correlation between two feature maps
        corr = torch.abs(torch.corrcoef(x)[0][1])
        # Check if tensor is on GPU
        if corr.is_cuda:
            # Move tensor from GPU to CPU and transform to NumPy
            corr = corr.cpu().detach().numpy()
        else:
            corr.numpy()

        return corr

    def feature_maps_correlation(
            self,
            model1_name: str,
            model2_name: str,
            model1_layer_name: str,
            model2_layer_name: str,
            model1_feature_map,
            model2_feature_map,
            n_maps1: int,
            n_maps2: int
    ):
        """Function to compute the correlation between feature maps of two layers from different models.

        Parameters:
        -----------
        model1_name : str
            Name of the first model.
        model2_name : str
            Name of the second model.
        model1_layer_name : str
            Name of the layer of the first model to compute the feature maps correlation.
        model2_layer_name : str
            Name of the layer of the second model to compute the feature maps correlation.
        model1_feature_map : torch.Tensor
            3D tensor with shape (batch_size, height, width) containing the feature maps of the chosen layer
            from the first model.
        model2_feature_map : torch.Tensor
            3D tensor with shape (batch_size, height, width) containing the feature maps of the chosen layer
            from the second model.
        n_maps1 : int
            Number of feature maps of the chosen layer from the first model.
        n_maps2 : int
            Number of feature maps of the chosen layer from the second model.

        Returns:
        --------
        pandas.DataFrame
        Dataframe containing the correlation values between each pair of feature maps, along with metadata such
        as the ids and names of the models and layers, and flags indicating if any of the feature maps contain only
        zeros.

        """
        fm_correlation = []

        # Create Object to progress bar
        prog = pyprog.ProgressBar(" ", "", n_maps1)
        # Print Task name
        print(
            f"""Computing feature maps correlation: {model1_name + '_' + model1_layer_name} - {model2_name + '_' + model2_layer_name} \n"""
        )
        # Update Progress Bar
        prog.update()

        for map_idx1 in range(n_maps1):
            # Reshaping it into a one-dimensional tensor
            layer_1_map_1d = model1_feature_map[map_idx1].flatten()

            for map_idx2 in range(n_maps2):
                # Reshaping it into a one-dimensional tensor
                layer_2_map_1d = model2_feature_map[map_idx2].flatten()

                zero_flag_fm_1, zero_flag_fm_2 = check_if_map_is_zero(layer_1_map_1d, layer_2_map_1d)

                if (zero_flag_fm_1 == 1) and (zero_flag_fm_2 == 1):
                    corr = 1.0
                elif (zero_flag_fm_1 == 1) and (zero_flag_fm_2 == 0):
                    corr = 0.0
                elif (zero_flag_fm_1 == 0) and (zero_flag_fm_2 == 1):
                    corr = 0.0
                else:
                    zero_flag_fm_1 = 0
                    zero_flag_fm_2 = 0

                    # Calculate Pearson correlation between two feature maps
                    corr = self.pearson_correlation(layer_1_map_1d, layer_2_map_1d)

                # Rename models if they have the same name
                model1_name, model2_name = check_models_name(model1_name, model2_name)

                # Append data to dict
                fm_correlation.append({
                    model1_name + '_fm_id': map_idx1,
                    model2_name + '_fm_id': map_idx2,
                    'correlation': corr,
                    model1_name + '_zero_flag': zero_flag_fm_1,
                    model2_name + '_zero_flag': zero_flag_fm_2,
                    model1_name + '_layer': model1_layer_name,
                    model2_name + '_layer': model2_layer_name
                })

            # Set current status
            prog.set_stat(map_idx1 + 1)
            # Update Progress Bar
            prog.update()

        # Make the Progress Bar final
        prog.end()
        print('\n')

        # Convert the list to a dataframe
        fm_correlation = pd.DataFrame(fm_correlation)
        # Reset indexes
        fm_correlation.reset_index(drop=True, inplace=True)

        return fm_correlation

    def multiple_feature_maps_correlation(
            self,
            layers_metadata1: dict[str, Any],
            layers_metadata2: dict[str, Any],
            feature_list_model1: list[str],
            feature_list_model2: list[str]
    ) -> tuple[Union[Union[Series, DataFrame], Any], dict[str, Any]]:
        """Function to compute the correlation between feature maps of multiple layers from different models.

        """

        # A dict to store DataFrames of feature maps correlations
        feature_maps_corr, feature_maps_corr_dict = [], {}
        model1_name = layers_metadata1['model_name']
        model2_name = layers_metadata2['model_name']

        for idx1, layer_name1 in enumerate(layers_metadata1['layer_path']):
            n_maps1 = layers_metadata1['n_maps'][idx1]

            for idx2, layer_name2 in enumerate(layers_metadata2['layer_path']):
                n_maps2 = layers_metadata2['n_maps'][idx2]

                feature_maps_corr_dict[
                    f'{model1_name}_{model2_name}_{layer_name1}_{layer_name2}'] = self.feature_maps_correlation(
                    model1_name=model1_name,
                    model2_name=model2_name,
                    model1_layer_name=layer_name1,
                    model2_layer_name=layer_name2,
                    model1_feature_map=feature_list_model1[:][idx1],
                    model2_feature_map=feature_list_model2[:][idx2],
                    n_maps1=n_maps1,
                    n_maps2=n_maps2
                )

                feature_maps_corr.append(
                    feature_maps_corr_dict[f'{model1_name}_{model2_name}_{layer_name1}_{layer_name2}'])

        feature_maps_corr = pd.concat(feature_maps_corr)

        return feature_maps_corr, feature_maps_corr_dict

    def get_min_correlations(
            self,
            layers_metadata1: dict,
            layers_metadata2: dict,
            feature_maps_corr_dict: dict,
            max_corr_threshold: float = 1.0
    ) -> tuple[Union[Union[Series, DataFrame], Any], dict[str, Any]]:
        """Obtains the minimum correlation among all combinations of correlations between the feature maps of two models,
        given the metadata of two observed layers, and a dictionary with DataFrames that represent the correlation
        between the feature maps of the two models.

        Parameters:
        -----------
        layers_metadata1: dict
            A dictionary that contains the metadata for the first model's layer of interest.
            Example: {'layer_name': ['Conv1'], 'n_maps': [32]}

        layers_metadata2: dict
            A dictionary that contains the metadata for the second model's layer of interest.
            Example: {'layer_name': ['Conv2'], 'n_maps': [64]}

        feature_maps_corr_dict: dict
            A dictionary that contains DataFrames with the correlation values between the feature maps of the two models.
            Each key of the dictionary should represent a unique pair of layers, and each DataFrame should have the following
            columns: 'model1_fm_id', 'model2_fm_id', 'correlation', 'model1_layer_name', 'model2_layer_name'.

        max_corr_threshold: float (default 0.0)
            A threshold to filter the feature maps that have a correlation value less or equal this value.

        same_models: bool (default False)
            A boolean flag that indicates whether the two layers belong to the same model. If True, the function will not
            consider the correlation value of same feature maps from the same layer.

        Returns:
        --------
        min_feature_maps_corr: pd.DataFrame
            A DataFrame that contains the minimum correlation values between all combinations of feature maps of the two
            models. The DataFrame will have the following columns: 'model1_fm_id', 'model2_fm_id', 'correlation',
            'model1_layer_name', 'model2_layer_name'.
        """

        min_feature_maps_corr, min_feature_maps_dict = [], {}

        for i, key in enumerate(feature_maps_corr_dict.keys()):

            df_aux = pd.DataFrame()

            model1_fm_id_column = feature_maps_corr_dict[key].columns[0]
            model2_fm_id_column = feature_maps_corr_dict[key].columns[1]
            model1_layer_name_column = feature_maps_corr_dict[key].columns[-2]
            model2_layer_name_column = feature_maps_corr_dict[key].columns[-1]

            model1_layer_name = feature_maps_corr_dict[key][model1_layer_name_column][0]
            model2_layer_name = feature_maps_corr_dict[key][model2_layer_name_column][0]

            if layers_metadata1['n_maps'][0] >= layers_metadata2['n_maps'][0]:
                column_to_group = model1_fm_id_column
            else:
                column_to_group = model2_fm_id_column

            df_aux = feature_maps_corr_dict[key].copy()
            df_aux = df_aux[df_aux['correlation'] <= max_corr_threshold]

            # Drop the layer name columns
            df_aux.drop([model1_layer_name_column, model2_layer_name_column], axis=1, inplace=True)
            # Get the min correlation
            df_aux = df_aux.loc[
                df_aux
                .astype(float)
                .groupby(column_to_group)['correlation']
                .idxmin()
            ].reset_index(drop=True)

            df_aux[model1_layer_name_column] = model1_layer_name
            df_aux[model2_layer_name_column] = model2_layer_name

            min_feature_maps_corr.append(df_aux)
            min_feature_maps_dict[f'min_{key}'] = df_aux

        min_feature_maps_corr = pd.concat(min_feature_maps_corr)

        return min_feature_maps_corr, min_feature_maps_dict

    def get_max_correlations(
            self,
            layers_metadata1: dict,
            layers_metadata2: dict,
            feature_maps_corr_dict: dict,
            min_corr_threshold: float = 0.0,
            same_model: bool = False
    ) -> tuple[Union[Union[Series, DataFrame], Any], dict[str, Any]]:

        """Obtains the maximum correlation among all combinations of correlations between the feature maps of two models,
        given the metadata of two observed layers, and a dictionary with DataFrames that represent the correlation
        between the feature maps of the two models.

        Parameters:
        -----------
        layers_metadata1: dict
            A dictionary that contains the metadata for the first model's layer of interest.
            Example: {'layer_name': ['Conv1'], 'n_maps': [32]}

        layers_metadata2: dict
            A dictionary that contains the metadata for the second model's layer of interest.
            Example: {'layer_name': ['Conv2'], 'n_maps': [64]}

        feature_maps_corr_dict: dict
            A dictionary that contains DataFrames with the correlation values between the feature maps of the two models.
            Each key of the dictionary should represent a unique pair of layers, and each DataFrame should have the following
            columns: 'model1_fm_id', 'model2_fm_id', 'correlation', 'model1_layer_name', 'model2_layer_name'.

        min_corr_threshold: float (default 0.0)
            A threshold to filter the feature maps that have a correlation value greater or equal this value.

        same_model: bool (default False)
            A boolean flag that indicates whether the two layers belong to the same model. If True, the function will not
            consider the correlation value of same feature maps from the same layer.

        Returns:
        --------
        max_feature_maps_corr: pd.DataFrame
            A DataFrame that contains the maximum correlation values between all combinations of feature maps of the two
            models. The DataFrame will have the following columns: 'model1_fm_id', 'model2_fm_id', 'correlation',
            'model1_layer_name', 'model2_layer_name'.
        """

        max_feature_maps_corr, max_feature_maps_corr_dict = [], {}

        for i, key in enumerate(feature_maps_corr_dict.keys()):

            df_aux = pd.DataFrame()

            model1_fm_id_column = feature_maps_corr_dict[key].columns[0]
            model2_fm_id_column = feature_maps_corr_dict[key].columns[1]
            model1_layer_name_column = feature_maps_corr_dict[key].columns[-2]
            model2_layer_name_column = feature_maps_corr_dict[key].columns[-1]

            model1_layer_name = feature_maps_corr_dict[key][model1_layer_name_column][0]
            model2_layer_name = feature_maps_corr_dict[key][model2_layer_name_column][0]

            if layers_metadata1['n_maps'][0] >= layers_metadata2['n_maps'][0]:
                column_to_group = model1_fm_id_column
            else:
                column_to_group = model2_fm_id_column

            # Filter to avoid getting the correlation == 1 of the same feature map
            if same_model:
                df_aux = feature_maps_corr_dict[key].copy()
                df_aux = df_aux[df_aux[model1_fm_id_column] != df_aux[model2_fm_id_column]]
                df_aux = df_aux[df_aux['correlation'] >= min_corr_threshold]
            else:
                df_aux = feature_maps_corr_dict[key].copy()
                df_aux = df_aux[df_aux['correlation'] >= min_corr_threshold]

                # Drop the layer name columns
            df_aux.drop([model1_layer_name_column, model2_layer_name_column], axis=1, inplace=True)
            # Get the max correlation
            df_aux = df_aux.loc[
                df_aux
                .astype(float)
                .groupby(column_to_group)['correlation']
                .idxmax()
            ].reset_index(drop=True)

            df_aux[model1_layer_name_column] = model1_layer_name
            df_aux[model2_layer_name_column] = model2_layer_name

            max_feature_maps_corr.append(df_aux)
            max_feature_maps_corr_dict[f'max_{key}'] = df_aux

        max_feature_maps_corr = pd.concat(max_feature_maps_corr)

        return max_feature_maps_corr, max_feature_maps_corr_dict

    # TODO: documentar a função
    def get_correlation_stats(
            self,
            max_or_fm_corr_dict: dict,
            layers_metadata1: dict,
            layers_metadata2: dict,
            same_model=False
    ) -> tuple[Union[Union[Series, DataFrame], Any], dict[str, Any]]:

        """From the maximum or minimum correlation dict, this function calculates the stats of the most frequent
        correlations between feature maps of two models. The stats are: the number of feature maps, their mean,
        median and standard deviation.

        Parameters:
        -----------

        """
        stats_correlation, stats_correlation_dict = [], {}

        if same_model:
            model1_name, model2_name = check_models_name(layers_metadata1['model_name'], layers_metadata2['model_name'])
        else:
            model1_name, model2_name = layers_metadata1['model_name'], layers_metadata2['model_name']

        for i, key in enumerate(max_or_fm_corr_dict.keys()):

            if layers_metadata1['n_maps'][0] < layers_metadata2['n_maps'][0]:
                column_to_group = max_or_fm_corr_dict[key].columns[0]
                model_layer = model1_name + '_layer'
                # TODO Generalizar a obtenção do nmaps para qualquer layer com diferentes nmaps
                n_maps = layers_metadata1['n_maps'][0]
            else:
                column_to_group = max_or_fm_corr_dict[key].columns[1]
                model_layer = model2_name + '_layer'
                n_maps = layers_metadata1['n_maps'][0]

            df_aux = max_or_fm_corr_dict[key].copy()
            pivoted_stats = pd.pivot_table(
                df_aux,
                index=[column_to_group, model_layer],
                values=['correlation'],
                aggfunc=('count', 'mean', 'median', 'std')
            ).swaplevel(0, 1, axis=1)  # swap the two innermost levels of the index

            pivoted_stats.columns = pivoted_stats.columns.map('_'.join)
            pivoted_stats.reset_index(level=1, inplace=True)
            pivoted_stats.reset_index(level=0, inplace=True)
            pivoted_stats['count_freq(%)'] = round(100 * pivoted_stats['count_correlation'] / n_maps, 4)
            pivoted_stats.sort_values(by=[model_layer, 'count_correlation'], ascending=(True, False), inplace=True)
            pivoted_stats.reset_index(drop=True, inplace=True)
            pivoted_stats.replace(np.nan, 0, inplace=True)

            stats_correlation.append(pivoted_stats)
            stats_correlation_dict[f'stats_{key}'] = pivoted_stats

        stats_correlation = pd.concat(stats_correlation)

        return stats_correlation, stats_correlation_dict

    def correlation_pipeline(
            self,
            img: torch.Tensor,
            layers_list1: list,
            layers_list2: list,
            model1: torch.nn.Module,
            model2: torch.nn.Module,
            model1_name: str,
            model2_name: str,
            same_model=False,
            min_corr_threshold: float = 0.0,
            max_corr_threshold: float = 1.0,
            save_path: str = None,
            condensed_files: bool = True,
            file_type: str = 'csv',
            device: str = 'cuda',
            memory_cleaning: bool = False
    ) -> tuple[Union[Union[Series, DataFrame], Any], dict[str, Any]]:
        """
        This function is the pipeline to extract the correlation between feature maps of two models.
        It is possible to extract the correlation between feature maps of the same model or between
        feature maps of different models.

        Parameters:
        -----------
        img: torch.Tensor
            The image to be used to extract the feature maps.
        layers_list1: list
            The list of layers to be used to extract the feature maps from the first model.
        layers_list2: list
            The list of layers to be used to extract the feature maps from the second model.
        model1: torch.nn.Module
            The first model to be used to extract the feature maps.
        model2: torch.nn.Module
            The second model to be used to extract the feature maps.
        model1_name: str
            The name of the first model.
        model2_name: str
            The name of the second model.
        same_model: bool
            If True, the feature maps will be extracted from the same model. If False, the feature maps will be
            extracted from different models.
        min_corr_threshold: float
            The minimum correlation threshold to be used to filter the feature maps.
        max_corr_threshold: float
            The maximum correlation threshold to be used to filter the feature maps.
        save_path: str
            The path to save the results.
        condensed_files: bool
            If True, the results will be saved in a single file. If False, the results will be saved in
            multiple files.
        file_type: str
            The type of file to save the results. It can be 'csv' or 'json'.
        device: str
            The device to be used to extract the feature maps. It can be 'cpu' or 'cuda'.
        memory_cleaning: bool
            If True, the memory will be cleaned after each layer. If False, the memory will not be cleaned.

        Returns:
        --------
        max_corr_dict: dict
            A dictionary with the maximum correlation between feature maps of two models.
        max_or_fm_corr_dict: dict
            A dictionary with the maximum correlation between feature maps of two models or the feature maps
            with the highest correlation.
        stats_correlation: pandas.DataFrame
            A pandas DataFrame with the statistics of the correlation between feature maps of two models.
        stats_correlation_dict: dict
            A dictionary with the statistics of the correlation between feature maps of two models.
      """
        # Initialize ExtractCnnLayers class
        erl_model1 = ExtractCnnLayers(model=model1, model_name=model1_name)
        erl_model2 = ExtractCnnLayers(model=model2, model_name=model2_name)

        # Get layers metadata from model
        layers_metadata_model1 = erl_model1.get_layers(layers_paths=layers_list1)
        layers_metadata_model2 = erl_model2.get_layers(layers_paths=layers_list2)

        # Initialize ExtractCnnMaps
        erm_model1 = ExtractCnnMaps(model=model1, dataset=None, image=img, device=device)
        erm_model2 = ExtractCnnMaps(model=model2, dataset=None, image=img, device=device)

        # Extract feature maps from layers
        fm_list_model1 = erm_model1.get_multiple_feature_maps(layers=layers_metadata_model1['layer'])
        fm_list_model2 = erm_model2.get_multiple_feature_maps(layers=layers_metadata_model2['layer'])

        # Compute correlation between features maps of two models
        self.fm_corr, self.fm_corr_dict = self.multiple_feature_maps_correlation(
            layers_metadata1=layers_metadata_model1,
            layers_metadata2=layers_metadata_model2,
            feature_list_model1=fm_list_model1,
            feature_list_model2=fm_list_model2
        )
        # Compute the maximum correlation between features maps of different layers
        self.max_fm_corr, self.max_fm_corr_dict = self.get_max_correlations(
            layers_metadata1=layers_metadata_model1,
            layers_metadata2=layers_metadata_model2,
            feature_maps_corr_dict=self.fm_corr_dict,
            min_corr_threshold=min_corr_threshold,
            same_model=same_model
        )
        # Compute the minimum correlation between features maps of different layers
        self.min_fm_corr, self.min_fm_corr_dict = self.get_min_correlations(
            layers_metadata1=layers_metadata_model1,
            layers_metadata2=layers_metadata_model2,
            feature_maps_corr_dict=self.fm_corr_dict,
            max_corr_threshold=max_corr_threshold
        )
        # Compute statistics for the maximum correlations
        self.stats_max_corr, self.stats_max_corr_dict = self.get_correlation_stats(
            max_or_fm_corr_dict=self.max_fm_corr_dict,
            layers_metadata1=layers_metadata_model1,
            layers_metadata2=layers_metadata_model2,
            same_model=same_model
        )
        # Compute statistics for the minimum correlations
        self.stats_min_corr, self.stats_min_corr_dict = self.get_correlation_stats(
            max_or_fm_corr_dict=self.min_fm_corr_dict,
            layers_metadata1=layers_metadata_model1,
            layers_metadata2=layers_metadata_model2,
            same_model=same_model
        )

        if save_path is not None:
            self.save_all_files(
                save_path=save_path,
                layers_metadata1=layers_metadata_model1,
                layers_metadata2=layers_metadata_model2,
                file_type=file_type,
                condensed_files=condensed_files
            )

        if memory_cleaning:
            del fm_list_model1, fm_list_model2
            gc.collect()
            torch.cuda.empty_cache()
        else:
            return self.fm_corr_dict, self.max_fm_corr_dict, self.min_fm_corr_dict, self.stats_max_corr_dict, self.stats_min_corr_dict

    def save_all_files(
            self,
            save_path: str,
            layers_metadata1: dict,
            layers_metadata2: dict,
            file_type: str = 'csv',
            condensed_files: bool = True
    ) -> None:
        """Save all results in a single file or in multiple files.

        Parameters:
        -----------
        save_path: str
            The path to save the results.
        layers_metadata1: dict
            A dictionary with the metadata of the layers of the first model.
        layers_metadata2: dict
            A dictionary with the metadata of the layers of the second model.
        file_type: str
            The type of file to save the results. It can be 'csv' or 'json'.
        condensed_files: bool
            If True, the results will be saved in a single file. If False, the results will be saved in multiple files.

        Returns:
        --------
        None

      """

        # TODO: Check if the directory is valid. If not, create the directory.
        if condensed_files:
            model1_name = layers_metadata1['model_name']
            model2_name = layers_metadata2['model_name']
            # Save all condensed dataframes as json
            if file_type == 'json':
                self.fm_corr.to_json(
                    path_or_buf=f'{save_path}/{model1_name}_vs_{model2_name}_fm_corr.json',
                    orient="index"
                )
                self.max_fm_corr.to_json(
                    path_or_buf=f'{save_path}/max_{model1_name}_vs_{model2_name}_fm_corr.json',
                    orient="index"
                )
                self.min_fm_corr.to_json(
                    path_or_buf=f'{save_path}/min_{model1_name}_vs_{model2_name}_fm_corr.json',
                    orient="index"
                )
                self.stats_max_corr.to_json(
                    path_or_buf=f'{save_path}/stats_max_{model1_name}_vs_{model2_name}.json',
                    orient="index"
                )
                self.stats_min_corr.to_json(
                    path_or_buf=f'{save_path}/stats_min_{model1_name}_vs_{model2_name}.json',
                    orient="index"
                )
            # Save all condensed dataframes as CSV
            elif file_type == 'csv':
                self.fm_corr.to_csv(
                    path_or_buf=f'{save_path}/{model1_name}_vs_{model2_name}_fm_corr.csv',
                    index=False
                )
                self.max_fm_corr.to_csv(
                    path_or_buf=f'{save_path}/max_{model1_name}_vs_{model2_name}_fm_corr.csv',
                    index=False
                )
                self.min_fm_corr.to_csv(
                    path_or_buf=f'{save_path}/min_{model1_name}_vs_{model2_name}_fm_corr.csv',
                    index=False
                )
                self.stats_max_corr.to_csv(
                    path_or_buf=f'{save_path}/stats_max_{model1_name}_vs_{model2_name}.csv',
                    index=False
                )
                self.stats_min_corr.to_csv(
                    path_or_buf=f'{save_path}/stats_min_{model1_name}_vs_{model2_name}.csv',
                    index=False
                )

        if condensed_files is False:
            # Save stats_max_corr_dict's dataframes as json or CSV
            if file_type == 'json':
                # Extract the keys from the dictionary and save the dataframes as json
                for (key1, key2, key3, key4, key5) in zip(self.fm_corr_dict.keys(),
                                                          self.max_fm_corr_dict.keys(),
                                                          self.min_fm_corr_dict.keys(),
                                                          self.stats_max_corr_dict.keys(),
                                                          self.stats_min_corr_dict.keys()
                                                          ):
                    self.fm_corr_dict[key1].to_json(
                        path_or_buf=f'{save_path}/{key1}.json',
                        orient="index"
                    )
                    self.max_fm_corr_dict[key2].to_json(
                        path_or_buf=f'{save_path}/{key2}.json',
                        orient="index"
                    )
                    self.min_fm_corr_dict[key3].to_json(
                        path_or_buf=f'{save_path}/{key3}.json',
                        orient="index"
                    )
                    self.stats_max_corr_dict[key4].to_json(
                        path_or_buf=f'{save_path}/{key4}.json',
                        orient="index"
                    )
                    self.stats_min_corr_dict[key5].to_json(
                        path_or_buf=f'{save_path}/{key5}.json',
                        orient="index"
                    )
            # Extract the keys from the dictionary and save the dataframes as csv
            elif file_type == 'csv':
                for (key1, key2, key3, key4, key5) in zip(self.fm_corr_dict.keys(),
                                                          self.max_fm_corr_dict.keys(),
                                                          self.min_fm_corr_dict.keys(),
                                                          self.stats_max_corr_dict.keys(),
                                                          self.stats_min_corr_dict.keys()
                                                          ):
                    self.fm_corr_dict[key1].to_csv(
                        path_or_buf=f'{save_path}/{key1}.csv',
                        sep=',', index=False
                    )
                    self.max_fm_corr_dict[key2].to_csv(
                        path_or_buf=f'{save_path}/{key2}.csv',
                        sep=',', index=False
                    )
                    self.min_fm_corr_dict[key3].to_csv(
                        path_or_buf=f'{save_path}/{key3}.csv',
                        sep=',', index=False
                    )
                    self.stats_max_corr_dict[key4].to_csv(
                        path_or_buf=f'{save_path}/{key4}.csv',
                        sep=',', index=False
                    )
