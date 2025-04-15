# References:
# https://github.com/facebookresearch/higher/blob/main/examples/support/omniglot_loaders.py
# https://github.com/dragen1860/MAML-Pytorch

from data.dataset_loader.utils import divide_array, LocalDataset
import os.path
import torch
import numpy as np
from torch.utils.data import Dataset
from global_config import ROOT_DIRECTORY, PROJECTS_DIRECTORY
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def distribution_visualization_wireless_channel_distributions(wireless_channel_distribution_dataset, device="cpu"):
    datasets, labels, datasets_spatial_frequency_domain = (tensor.to(device) for tensor in wireless_channel_distribution_dataset[:])
    datasets = datasets #- 0.5

    datasets = torch.view_as_complex(datasets.permute(0, 2, 3, 1).contiguous()) # Centering again, converting to complex domain
    datasets_sf = torch.view_as_complex(datasets_spatial_frequency_domain.permute(0, 2, 3, 1).contiguous()) #datasets_spatial_frequency_domain[:, 0] + 1j * datasets_spatial_frequency_domain[:, 1]

    # Zero-padding for FFT
    zero_padding = torch.zeros(datasets.shape[0], 32, 256 - 32, dtype=torch.complex64, device=device)
    input_tensor = torch.cat((datasets, zero_padding), dim=2)

    # Perform 2D FFT
    H_SF_hat = torch.fft.fft2(input_tensor)

    num_datasets = 8
    dataset_size = datasets.shape[0] // num_datasets

    fig, ax = plt.subplots(figsize=(8, 6))

    # Loop through datasets and compute CDF
    for i in range(num_datasets):
        # Extract absolute channel gains
        channel_magnitudes = torch.mean(torch.abs(datasets[i * dataset_size: (i + 1) * dataset_size]), dim=[1,2]).cpu().numpy()
        sorted_data = np.sort(channel_magnitudes)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=f'Distribution {i + 1}')
        ax.set_xlabel('power')
        ax.set_ylabel('CDF')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compute mean power
    psd_data = [torch.mean(torch.abs(datasets[i * dataset_size: (i + 1) * dataset_size]), dim=0).cpu().numpy()
                for i in range(num_datasets)]

    psd_data_var = [torch.var(torch.abs(datasets[i * dataset_size: (i + 1) * dataset_size]), dim=0).cpu().numpy()
                for i in range(num_datasets)]

    # Create subplots for 3D plots
    fig, axes = plt.subplots(2, num_datasets // 2, figsize=(16, 10), subplot_kw={'projection': '3d'})

    # Find global vmin and vmax for consistent color scaling
    vmin = np.min([psd.min() for psd in psd_data])
    vmax = np.max([psd.max() for psd in psd_data])

    # Plot each dataset in a 3D surface plot
    X, Y = np.meshgrid(np.arange(32), np.arange(32))

    for i in range(num_datasets):
        ax = axes[i % 2, i // 2]

        # Normalize variance for color mapping
        var_norm = (psd_data_var[i] - np.min(psd_data_var[i])) / (np.max(psd_data_var[i]) - np.min(psd_data_var[i]))

        # Create surface plot
        surf = ax.plot_surface(X, Y, psd_data[i], facecolors=plt.cm.viridis(var_norm), rstride=1, cstride=1, alpha=0.9)

        ax.set_title(f"Dataset {i + 1}")
        ax.set_xlabel("Delay")
        ax.set_ylabel("Angular")
        ax.set_zlabel("Mean Power")

    plt.show()

def sample_visualization_wireless_channel_distributions(wireless_channel_distribution_dataset, device="cpu", indices=None):
    datasets, labels, datasets_spatial_frequency_domain = (tensor.to(device) for tensor in wireless_channel_distribution_dataset[:])
    datasets = datasets #- 0.5

    datasets = torch.view_as_complex(datasets.permute(0, 2, 3, 1).contiguous()) # Centering again, converting to complex domain
    datasets_sf = torch.view_as_complex(datasets_spatial_frequency_domain.permute(0, 2, 3, 1).contiguous()) #datasets_spatial_frequency_domain[:, 0] + 1j * datasets_spatial_frequency_domain[:, 1]

    # Zero-padding for FFT
    zero_padding = torch.zeros(datasets.shape[0], 32, 256 - 32, dtype=torch.complex64, device=device)
    input_tensor = torch.cat((datasets, zero_padding), dim=2)

    # Perform 2D FFT
    H_SF_hat = torch.fft.fft2(input_tensor)

    # Define the distribution ranges
    num_distributions = 8
    samples_per_dist = 10_000  # Each distribution contains 10,000 samples

    num_datasets = 8
    dataset_size = datasets.shape[0] // num_datasets

    for idx in indices:
        fig, axes = plt.subplots(4, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
        data_list = [datasets_sf[idx], datasets[idx], input_tensor[idx], H_SF_hat[idx]]
        titles = ['CSI (SF domain)', 'CSI (AD domain, cropped)', 'CSI (AD domain, with zero padding)', 'Recovered CSI (SF domain)']
        for ax, data, title in zip(axes, data_list, titles):
            img = ax.imshow(torch.abs(data).cpu().numpy(), cmap='viridis')
            ax.set_title(f'{title}, idx:{idx}')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()


class WirelessChannelDistributions(Dataset):
    def __init__(self, mode="train", normalization=True):
        datasets, datasets_spatial_frequency_domain, labels = [], [], []
        selected_scenarios = [1, 2, 3, 4, 5, 6, 7, 8]

        for scenario_idx in selected_scenarios:
            dataset_folder_path = os.path.join(PROJECTS_DIRECTORY, "dataset", "wireless_channel_distributions_dataset",
                                               "scenario_" + str(scenario_idx))
            dataset_path = os.path.join(dataset_folder_path, f"s{str(scenario_idx)}_{mode}.npy")

            dataset = np.load(dataset_path, allow_pickle=True).item()
            csi_sample_angular_delay = dataset["h"]
            csi_sample_spatial_frequency = dataset["h_spatial_frequency"]

            # Cropped Angular Delay Domain CSI
            csi_sample_angular_delay = np.stack([np.real(csi_sample_angular_delay),
                                                 np.imag(csi_sample_angular_delay)], axis=-1)
            csi_sample_angular_delay = np.transpose(csi_sample_angular_delay, (0, 3, 1, 2))
            if normalization:
                max_vals = np.max(np.abs(csi_sample_angular_delay), axis=(1, 2, 3), keepdims=True)
                csi_sample_angular_delay = csi_sample_angular_delay / (2 * max_vals) + 0.5

            # Spatial-Frequency Domain CSI
            csi_sample_spatial_frequency = np.stack([np.real(csi_sample_spatial_frequency),
                                                     np.imag(csi_sample_spatial_frequency)], axis=-1)
            csi_sample_spatial_frequency = np.transpose(csi_sample_spatial_frequency, (0, 3, 1, 2))
            if normalization:
                max_vals = np.max(np.abs(csi_sample_spatial_frequency), axis=(1, 2, 3), keepdims=True)
                csi_sample_spatial_frequency = csi_sample_spatial_frequency / (max_vals)

            label = scenario_idx * np.ones(shape=(len(csi_sample_angular_delay), 1))

            datasets.append(csi_sample_angular_delay)
            datasets_spatial_frequency_domain.append(csi_sample_spatial_frequency)
            labels.append(label)

        datasets = np.concatenate(datasets).astype(np.float32)
        datasets_spatial_frequency_domain = np.concatenate(datasets_spatial_frequency_domain).astype(np.float32)
        labels = np.concatenate(labels).astype(np.float32)
        print("dataset variance: {}".format(np.var(datasets)))

        self.imgs = torch.from_numpy(datasets)
        self.imgs_spatial_frequency = torch.from_numpy(datasets_spatial_frequency_domain)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx], self.imgs_spatial_frequency[idx]


def get_wireless_channel_distributions_local_datasets(config):
    train_dataset = WirelessChannelDistributions(mode="train")
    test_dataset = WirelessChannelDistributions(mode="test")
    train_subarrays = divide_array(np.arange(len(train_dataset)), n_clients=config["n_clients"])
    test_subarrays = divide_array(np.arange(len(test_dataset)), n_clients=config["n_clients"])
    local_train_datasets = [LocalDataset(train_dataset, indices, task=config["task"]) for indices in train_subarrays]
    local_test_datasets = [LocalDataset(test_dataset, indices, task=config["task"]) for indices in test_subarrays]

    return local_train_datasets, local_test_datasets

if __name__ == "__main__":
    dataset = WirelessChannelDistributions(normalization=False)
    _ = distribution_visualization_wireless_channel_distributions(wireless_channel_distribution_dataset=dataset, device="cpu")
    print("hello world!")

