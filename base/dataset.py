from torch.utils.data import Dataset

import os
from base.utils import load_single_pkl
from base.transforms3D import *

from torchvision.transforms import transforms
import numpy as np
import random
from operator import itemgetter
from sklearn.model_selection import KFold

class ABAW2_VA_Arranger(object):
    def __init__(self, dataset_path, window_length=300, hop_length=300, debug=0):
        self.dataset_path = os.path.join(dataset_path, "npy_data")
        self.dataset_info = load_single_pkl(directory=dataset_path, filename="dataset_info")
        self.mean_std_info = load_single_pkl(directory=dataset_path, filename="mean_std_dict")
        self.window_length = window_length
        self.hop_length = hop_length
        self.debug = debug

    @staticmethod
    def generate_partition_dict_for_cross_validation(partition_dict, fold):
        new_partition_dict = {'Train_Set': {}, 'Validation_Set': {}, 'Test_Set': {}}
        partition_pool = {**partition_dict['Train_Set'], **partition_dict['Validation_Set']}

        trials_of_train_set = list(partition_dict['Train_Set'].keys())
        trials_of_original_validate_set = list(partition_dict['Validation_Set'].keys())
        trials_of_putative_test_set = list(partition_dict['Test_Set'].keys())

        fold_0_trials = trials_of_train_set[slice(0, 70)]
        fold_1_trials = trials_of_train_set[slice(70, 140)]
        fold_2_trials = trials_of_train_set[slice(140, 210)]
        fold_3_trials = trials_of_train_set[slice(210, 280)]
        fold_4_trials = trials_of_train_set[slice(280, 351)]
        fold_5_trials = trials_of_original_validate_set

        fold_n_trials = [
            fold_0_trials,
            fold_1_trials,
            fold_2_trials,
            fold_3_trials,
            fold_4_trials,
            fold_5_trials
        ]

        fold_index = np.roll(np.arange(len(fold_n_trials)), fold)
        ordered_trials = list(itemgetter(*fold_index)(fold_n_trials))

        for nth_fold, trials_of_a_fold in enumerate(ordered_trials):
            for trial in trials_of_a_fold:
                partition = "Train_Set"
                if nth_fold == len(fold_n_trials) - 1:
                    partition = "Validation_Set"
                new_partition_dict[partition].update({trial: partition_pool[trial]})

        new_partition_dict['Test_Set'] = partition_dict['Test_Set']
        return new_partition_dict
    
    @staticmethod
    def split_train_test_trial(partition_dict, mode='Test', seed=0):
        random.seed(seed)
        
        # Train_Test를 나눌 때
        if mode == 'Test':
            new_partition_dict = {'Train_Set': {}, 'Test_Set': {}}
            
            # 전체 데이터로 모아서
            partition_dict['Train_Set'].update(partition_dict['Validation_Set'])
            partition_dict = partition_dict['Train_Set']
            
            # 20%만 테스트로 사용
            test = random.sample(partition_dict.keys(), int(len(partition_dict.keys()) * 0.2))
            train = [t for t in partition_dict.keys() if not t in test]
            
            for trial_train in train:
                new_partition_dict["Train_Set"].update({trial_train:partition_dict[trial_train]})
                
            for trial_test in test:
                new_partition_dict["Test_Set"].update({trial_test:partition_dict[trial_test]})
                
            return new_partition_dict
        
        # Train_Valid 나눌 때
        elif mode == 'Valid':
            new_partition_dict = {'Train_Set': {}, 'Validation_Set': {}}
            
            # 20%만 테스트로 사용
            valid = random.sample(partition_dict.keys(), int(len(partition_dict.keys()) * 0.2))
            train = [t for t in partition_dict.keys() if not t in valid]
            
            for trial_train in train:
                new_partition_dict["Train_Set"].update({trial_train:partition_dict[trial_train]})
                
            for trial_valid in valid:
                new_partition_dict["Validation_Set"].update({trial_valid:partition_dict[trial_valid]})
                
            partition_dict = new_partition_dict
                
            return partition_dict            
            
    
    def custom_sampled_list(self, sampled_list, partition, trial, length):
        start = 0
        end = start + self.window_length

        if end < length:
            # Windows before the last one
            while end < length:
                indices = np.arange(start, end)
                path = os.path.join(self.dataset_path, trial)
                sampled_list[partition].append([path, trial, indices, length])
                start = start + self.hop_length
                end = start + self.window_length

            # The last window ends with the trial length, which ensures that no data are wasted.
            start = length - self.window_length
            end = length
            indices = np.arange(start, end)
            path = os.path.join(self.dataset_path, trial)
            sampled_list[partition].append([path, trial, indices, length])
        else:
            end = length
            indices = np.arange(start, end)
            path = os.path.join(self.dataset_path, trial)
            sampled_list[partition].append([path, trial, indices, length])
            
        return sampled_list
    

    def resample_according_to_window_and_hop_length(self, seed=0):
        partition_dict = self.dataset_info['partition']

        # partition_dict = self.generate_partition_dict_for_cross_validation(partition_dict, fold)
        
        # train / test
        partition_dict = self.split_train_test_trial(partition_dict, mode='Test', seed=seed)
        partition_dict_test = partition_dict['Test_Set']
        
        # train / valid 
        partition_dict = self.split_train_test_trial(partition_dict['Train_Set'], mode='Valid', seed=seed)
        partition_dict['Test_Set'] = partition_dict_test

        sampled_list_dict = {'Train_Set': [], 'Validation_Set': [], 'Test_Set':[]}
        for train_trials in partition_dict['Train_Set']:
            trial_count = 0
            partition = "Train_Set"
            length = partition_dict['Train_Set'][train_trials]
            self.custom_sampled_list(sampled_list_dict, partition, train_trials, length)
            trial_count += 1
            if self.debug and trial_count >= self.debug:
                break
            
        for valid_trials in partition_dict['Validation_Set']:
            trial_count = 0
            partition = "Validation_Set"
            length = partition_dict['Validation_Set'][valid_trials]
            self.custom_sampled_list(sampled_list_dict, partition, valid_trials, length)
            trial_count += 1
            if self.debug and trial_count >= self.debug:
                break
            
        for test_trials in partition_dict['Test_Set']:
            trial_count = 0
            partition = "Test_Set"
            length = partition_dict['Test_Set'][test_trials]
            self.custom_sampled_list(sampled_list_dict, partition, test_trials, length)
            trial_count += 1
            if self.debug and trial_count >= self.debug:
                break                        
            
        return sampled_list_dict
        
        # sampled_list = {'Train_Set': [], 'Validation_Set': [], 'Test_Set': []}

        # for partition, trials in partition_dict.items():
        #     trial_count = 0
        #     for trial, length in trials.items():

        #         start = 0
        #         end = start + self.window_length

        #         if end < length:
        #             # Windows before the last one
        #             while end < length:
        #                 indices = np.arange(start, end)
        #                 path = os.path.join(self.dataset_path, trial)
        #                 sampled_list[partition].append([path, trial, indices, length])
        #                 start = start + self.hop_length
        #                 end = start + self.window_length

        #             # The last window ends with the trial length, which ensures that no data are wasted.
        #             start = length - self.window_length
        #             end = length
        #             indices = np.arange(start, end)
        #             path = os.path.join(self.dataset_path, trial)
        #             sampled_list[partition].append([path, trial, indices, length])
        #         else:
        #             end = length
        #             indices = np.arange(start, end)
        #             path = os.path.join(self.dataset_path, trial)
        #             sampled_list[partition].append([path, trial, indices, length])

        #         trial_count += 1

        #         if self.debug and trial_count >= self.debug:
        #             break

        # return sampled_list


class ABAW2_VA_Dataset(Dataset):
    def __init__(self, data_list, time_delay=0, emotion="both", head="multi_head", modality = ['frame'], window_length=300, mode='train', fold=0, mean_std_info=None):
        self.data_list = data_list
        self.time_delay = time_delay
        self.emotion = emotion
        self.head = head
        self.modality = modality
        self.window_length = window_length
        self.mode = mode
        self.partition = self.get_partition_name_from_mode()
        self.fold = fold
        self.mean_std_info = mean_std_info
        self.get_3D_transforms()

    def get_partition_name_from_mode(self):
        if self.mode == "train":
            partition = "Train_Set"
        elif self.mode == "validate":
            partition = "Validation_Set"
        elif self.mode == "test":
            partition = "Test_Set"
        else:
            raise ValueError("Unknown partition!")

        return partition

    def get_3D_transforms(self):
        normalize = GroupNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if self.mode == 'train':
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupRandomCrop(48, 40),
                    GroupRandomHorizontalFlip(),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize
                ])

        if self.mode != 'train':
            if "frame" in self.modality:
                self.image_transforms = transforms.Compose([
                    GroupNumpyToPILImage(0),
                    GroupCenterCrop(40),
                    Stack(),
                    ToTorchFormatTensor(),
                    normalize 
                ])

        if "mfcc" in self.modality:
            try:
                self.mfcc_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean_std_info["mfcc"][self.fold]['Train_Set']['mean'],
                                        std=self.mean_std_info["mfcc"][self.fold]['Train_Set']['std']),
                ])
            except:
                self.mfcc_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean_std_info["mfcc"][self.fold]['Validation_Set']['mean'],
                                        std=self.mean_std_info["mfcc"][self.fold]['Validation_Set']['std']),
                ])
                

        if "vggish" in self.modality:
            try:
                self.vggish_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean_std_info["vggish"][self.fold]['Train_Set']['mean'],
                                        std=self.mean_std_info["vggish"][self.fold]['Train_Set']['std']),
                ])
            except:
                self.vggish_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean_std_info["vggish"][self.fold]['Validation_Set']['mean'],
                                        std=self.mean_std_info["vggish"][self.fold]['Validation_Set']['std']),
                ])                
                
        torch.cuda.empty_cache()


    @staticmethod
    def load_data(directory, indices, filename):
        filename = os.path.join(directory, filename)
        frames = np.load(filename, mmap_mode='c')[indices]
        return frames

    def __getitem__(self, index):
        path = self.data_list[index][0]
        trial = self.data_list[index][1]
        indices = self.data_list[index][2]
        length = self.data_list[index][3]
        labels = 0
        features = {}

        if "frame" in self.modality:

            if length < self.window_length:
                frames = np.zeros((self.window_length, 48, 48, 3), dtype=np.int16)
                frames[indices] = self.load_data(path, indices, "frame.npy")
            else:
                frames = self.load_data(path, indices, "frame.npy")

            frames = self.image_transforms(frames)
            features.update({'frame': frames})

        if "mfcc" in self.modality:
            if length < self.window_length:
                mfcc = np.zeros((self.window_length, 39), dtype=np.float32)
                mfcc[indices] = self.load_data(path, indices, "mfcc.npy")
            else:
                mfcc = self.load_data(path, indices, "mfcc.npy").astype(np.float32)
            mfcc = self.mfcc_transforms(mfcc)
            features.update({'mfcc': mfcc})

        if "vggish" in self.modality:
            if length < self.window_length:
                vggish = np.zeros((self.window_length, 128), dtype=np.float32)
                vggish[indices] = self.load_data(path, indices, "vggish.npy")
            else:
                vggish = self.load_data(path, indices, "vggish.npy").astype(np.float32)
            # vggish = (vggish - (-0.105)) / 0.4476
            vggish = self.vggish_transforms(vggish)
            features.update({'vggish': vggish})

        if self.mode != "test":
            if length < self.window_length:
                labels = np.zeros((self.window_length, 2), dtype=np.float32)
                labels[indices] = self.load_data(path, indices, "label.npy")
            else:
                labels = self.load_data(path, indices, "label.npy")

            if self.head == "single-headed":
                if self.emotion == "arousal": #Arousal
                    labels = labels[:, 1][:, np.newaxis]
                elif self.emotion == "valence": # Valence
                    labels = labels[:, 0][:, np.newaxis]
                else:
                    raise ValueError("Unsupported emotional dimension for continuous labels!")

            # Deal with the label delay by making the time_delay-th label point as the 1st point and shift other points accordingly.
            # The previous last point is copied for time_delay times.
            labels = np.concatenate(
                (labels[self.time_delay:, :],
                 np.repeat(labels[-1, :][np.newaxis], repeats=self.time_delay, axis=0)), axis=0)

        if len(indices) < self.window_length:
            indices = np.arange(self.window_length)

        return features, labels, trial, length, indices

    def __len__(self):
        return len(self.data_list)

