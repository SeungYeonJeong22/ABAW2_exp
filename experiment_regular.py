import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
from torch import nn
from torch.utils.data.distributed import DistributedSampler

# ABAW2
from base.utils import detect_device, select_gpu, set_cpu_thread
from configs import config_processing as config
from model.model import my_2d1d, my_2d1ddy
from model.model2 import TwoStreamAuralVisualModel, CAM
from base.dataset import ABAW2_VA_Arranger, ABAW2_VA_Dataset
from base.checkpointer import Checkpointer
from base.parameter_control import ParamControl
from base.trainer import ABAW2Trainer
import os

#JCA
from base.dataset_model2 import JCA_VA_Arranger, JCA_VA_Dataset

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred, weights=None):
        # pred = torch.tanh(pred)
        gold_mean = torch.mean(gold, 1, keepdim=True, out=None)
        pred_mean = torch.mean(pred, 1, keepdim=True, out=None)
        covariance = (gold - gold_mean) * (pred - pred_mean)
        gold_var = torch.var(gold, 1, keepdim=True, unbiased=True, out=None)
        pred_var = torch.var(pred, 1, keepdim=True, unbiased=True, out=None)
        ccc = 2. * covariance / (
                (gold_var + pred_var + torch.mul(gold_mean - pred_mean, gold_mean - pred_mean)) + 1e-08)
        ccc_loss = 1. - ccc

        if weights is not None:
            ccc_loss *= weights

        return torch.mean(ccc_loss)


class Experiment(object):
    def __init__(self, args, fold):
        self.args = args
        self.experiment_name = args.experiment_name
        self.dataset_path = args.dataset_path
        self.model_load_path = args.model_load_path
        self.model_save_path = args.model_save_path
        self.resume = args.resume
        self.debug = args.debug
        self.config = config
        
        print("self.experiment_name : " ,self.experiment_name) 

        self.gpu = args.gpu
        self.cpu = args.cpu
        # If the code is to run on high-performance computer, which is usually not
        # available to specify gpu index and cpu threads, then set them to none.
        if self.args.high_performance_cluster:
            self.gpu = None
            self.cpu = None

        self.stamp = args.stamp

        self.head = "single-headed"
        if args.head == "mh":
            self.head = "multi-headed"

        self.train_emotion = args.train_emotion

        self.emotion_dimension = self.get_train_emotion(args.train_emotion, args.head)
        self.modality = args.modality

        self.backbone_state_dict = args.backbone_state_dict
        self.backbone_mode = args.backbone_mode

        self.cnn1d_embedding_dim = args.cnn1d_embedding_dim
        self.cnn1d_channels = args.cnn1d_channels
        self.cnn1d_kernel_size = args.cnn1d_kernel_size
        self.cnn1d_dropout = args.cnn1d_dropout
        self.cnn1d_attention = args.cnn1d_attention
        self.lstm_embedding_dim = args.lstm_embedding_dim
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.lstm_dropout = args.lstm_dropout

        self.cross_validation = args.cross_validation
        # self.folds_to_run = args.folds_to_run
        self.folds_to_run = fold
        if not self.cross_validation:
            self.folds_to_run = [0]

        self.milestone = args.milestone
        self.learning_rate = args.learning_rate
        self.min_learning_rate = args.min_learning_rate
        self.early_stopping = args.early_stopping
        self.patience = args.patience
        self.time_delay = args.time_delay
        self.num_epochs = args.num_epochs
        self.min_num_epochs = args.min_num_epochs
        self.factor = args.factor

        self.window_length = args.window_length
        self.hop_length = args.hop_length
        self.batch_size = args.batch_size

        self.metrics = args.metrics
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.load_best_at_each_epoch = args.load_best_at_each_epoch

        self.save_plot = args.save_plot
        self.device = self.init_device()
        
        self.optim = args.optim
        
        # if "jca" in self.experiment_name:
        #     ## JCA
        #     self.root="../data/Affwild2/resized_cropped_aligned_images_224_224"
        #     self.fileList="../data/Affwild2/annotations/preprocessed_VA_annotations/Train_Set/"
        #     self.audList="../data/Affwild2/wav/"
        #     self.length=self.args.seq_length
        #     self.flag=self.args.flag
        #     self.stride=self.args.stride 
        #     self.dilation = self.args.dilation
        #     self.subseq_length = self.args.subseq_length        
        
        #     self.fusion_model = CAM().cuda()
            
        from datetime import datetime

        now = datetime.now()
        now_date = now.strftime('%Y%m%d_%H%M%S')
        
        

        self.model_name = now_date + "_" + self.experiment_name + "_" + args.model_name + "_" + self.modality[
            0] + "_" + self.train_emotion + "_" + args.head + "_bs_" + str(self.batch_size) + "_lr_" + str(
            self.learning_rate) + "_mlr_" + str(self.min_learning_rate) + "_" + self.optim + '_' + self.stamp


    def init_arranger(self, fold=0):
        self.init_random_seed()
        if self.experiment_name in "ABAW2":
            arranger = ABAW2_VA_Arranger(self.dataset_path, window_length=self.window_length, hop_length=self.hop_length,
                                        debug=self.debug)

            # For fold = 0, it is the original partition.
            data_dict = arranger.resample_according_to_window_and_hop_length(fold)
            # tra_val_fold, test_set = arranger.resample_according_to_window_and_hop_length(fold)
            
        elif self.experiment_name in "jca":
            arranger = JCA_VA_Arranger(self.dataset_path, window_length=self.window_length, hop_length=self.hop_length,
                                        debug=self.debug)

            # For fold = 0, it is the original partition.
            tra_val_fold = arranger.resample_according_to_window_and_hop_length(fold)
            
        # return tra_val_fold, test_set, arranger.mean_std_info
        return data_dict, arranger.mean_std_info
        

    # def init_dataloader(self, data_dict, mean_std_info, mode='train', fold=4):
    def init_dataloader(self, fold=4):
        self.init_random_seed()
        if self.experiment_name in "ABAW2":
            arranger = ABAW2_VA_Arranger(self.dataset_path, window_length=self.window_length, hop_length=self.hop_length,
                                        debug=self.debug)

            # For fold = 0, it is the original partition.
            data_dict = arranger.resample_according_to_window_and_hop_length(fold)
            random.shuffle(data_dict['Train_Set'])
            train_dataset = ABAW2_VA_Dataset(data_dict['Train_Set'], time_delay=self.time_delay, emotion=self.train_emotion,
                                            head=self.head, modality=self.modality,
                                            mode='train', fold=fold, mean_std_info=arranger.mean_std_info)
            self.init_random_seed()
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=self.batch_size, shuffle=False)

            validate_dataset = ABAW2_VA_Dataset(data_dict['Validation_Set'], time_delay=self.time_delay,
                                                emotion=self.train_emotion, modality=self.modality,
                                                head=self.head, mode='validate', fold=fold, mean_std_info=arranger.mean_std_info)
            validate_loader = torch.utils.data.DataLoader(
                dataset=validate_dataset, batch_size=self.batch_size, shuffle=False)

            dataloader_dict = {'train': train_loader, 'validate': validate_loader}
            
        # if mode=='test':
        #     test_dataset = ABAW2_VA_Dataset(data_dict, time_delay=self.time_delay,
        #                                         emotion=self.train_emotion, modality=self.modality,
        #                                         head=self.head, mode='test', fold=fold, mean_std_info=mean_std_info)
        #     test_loader = torch.utils.data.DataLoader(
        #         dataset=test_dataset, batch_size=self.batch_size, shuffle=False)            

        #     dataloader_dict = {'test':test_loader}
                

        # elif self.experiment_name in "jca":
        #     random.shuffle(data_dict['Train_Set'])
        #     train_dataset = JCA_VA_Dataset(data_dict['Train_Set'], time_delay=self.time_delay, emotion=self.train_emotion,
        #                                     head=self.head, modality=self.modality,
        #                                     mode='train', fold=fold, mean_std_info=mean_std_info)
            
        #     train_loader = torch.utils.data.DataLoader(
        #         dataset=train_dataset, batch_size=self.batch_size, shuffle=False)

        #     validate_dataset = JCA_VA_Dataset(data_dict['Validation_Set'], time_delay=self.time_delay,
        #                                         emotion=self.train_emotion, modality=self.modality,
        #                                         head=self.head, mode='validate', fold=fold, mean_std_info=mean_std_info)
        #     validate_loader = torch.utils.data.DataLoader(
        #         dataset=validate_dataset, batch_size=self.batch_size, shuffle=False)

        #     dataloader_dict = {'train': train_loader, 'validate': validate_loader}
            
        return dataloader_dict
    
    def mkdir_splitname(self, split_num):
        save_path = os.path.join(self.model_save_path, self.model_name, f"split_num_{split_num}")
        self.load_best_at_each_epoch = str(self.load_best_at_each_epoch)
        os.makedirs(save_path)
            
        checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")
        return save_path, checkpoint_filename


    def experiment(self, save_path, fold=0):
        criterion = CCCLoss()
        
        # save_path = os.path.join(self.model_save_path, self.model_name)
        self.load_best_at_each_epoch = str(self.load_best_at_each_epoch)
        # checkpoint_load_path = self.load_best_at_each_epoch + "/0"
        
        if self.resume==1:
            dirs_list = os.listdir(save_path)

            sorted_dirs = sorted(dirs_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

            save_path = os.path.join(save_path,f"{fold}")
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
        else:
            save_path = os.path.join(save_path,"0")
            os.makedirs(save_path)
            
        checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")
        print("checkpoint_filename : " , checkpoint_filename)

        model = self.init_model()
        model = nn.DataParallel(model, device_ids = [0,1]).cuda()
        # model = nn.DataParallel(model, device_ids = [1,0]).cuda()
        # model = nn.DataParallel(model).cuda()
        
        # dataloader_fold, dataloader_test, mean_std_info = self.init_arranger()
        data_dict, mean_std_info = self.init_arranger()
        
        # for  dataloader_dict in enumerate(dataloader_fold):
        #     save_path, checkpoint_filename = self.mkdir_splitname(split_num)
    
        # 파라미터가 다름
        if "ABAW2" in self.args.experiment_name:
            # ABAW2
            trainer = ABAW2Trainer(model, model_name=self.model_name, learning_rate=self.learning_rate,
                                min_learning_rate=self.min_learning_rate,
                                metrics=self.metrics, save_path=save_path, early_stopping=self.early_stopping,
                                train_emotion=self.train_emotion, patience=self.patience, factor=self.factor,
                                emotional_dimension=self.emotion_dimension, head=self.head, max_epoch=self.num_epochs,
                                load_best_at_each_epoch=self.load_best_at_each_epoch, window_length=self.window_length,
                                milestone=self.milestone, criterion=criterion, verbose=True, save_plot=self.save_plot,
                                optimizer=self.optim, device=self.device)

        ########
        # parameter_controller = ParamControl(trainer, gradual_release=self.gradual_release,
        #                                     release_count=self.release_count, backbone_mode=self.backbone_mode)

        # checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

        # if self.resume:
        #     trainer, parameter_controller = checkpoint_controller.load_checkpoint()
        # else:
        #     checkpoint_controller.init_csv_logger(self.args, self.config)


        # datatloader_fold: kfold 적용된 train/valid fold set
        # dataloader_test: kfold 적용안된 test set
        # dataloader_fold, dataloader_test, mean_std_info = self.init_arranger()

        # training
        # for  dataloader_dict in enumerate(dataloader_fold):
            
            parameter_controller = ParamControl(trainer, gradual_release=self.gradual_release,
                                                release_count=self.release_count, backbone_mode=self.backbone_mode)

            checkpoint_controller = Checkpointer(checkpoint_filename, trainer, parameter_controller, resume=self.resume)

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)            
            
            
            
            # dataloader_dict = self.init_dataloader(dataloader_dict, mean_std_info)
            dataloader_dict = self.init_dataloader(self.folds_to_run)
            
            if not trainer.fit_finished:
                trainer.fit(dataloader_dict, num_epochs=self.num_epochs, min_num_epochs=self.min_num_epochs,
                            save_model=True, parameter_controller=parameter_controller,
                            checkpoint_controller=checkpoint_controller)
                
                
        # # test
        # dataloader_test = self.init_dataloader(dataloader_test, mean_std_info, mode='test')
                
        # # kfold를 끝낸 후 테스트를 어떻게 불러내야 할지..
        # trainer.test(dataloader_test, num_epochs=self.num_epochs, min_num_epochs=self.min_num_epochs,
        #             save_model=True, parameter_controller=parameter_controller,
        #             checkpoint_controller=checkpoint_controller)

    def init_model(self):
        self.init_random_seed()

        if self.head == "multi-headed":
            output_dim = 2
        else:
            output_dim = 1

        if "2d1d" in self.model_name or "jca" in self.model_name:
            if len(self.modality) > 1:
                if "jca" in self.model_name:
                    model = TwoStreamAuralVisualModel(num_channels=4)
                else:
                    model = my_2d1ddy(backbone_state_dict=self.backbone_state_dict, backbone_mode=self.backbone_mode,
                                    embedding_dim=self.cnn1d_embedding_dim, channels=self.cnn1d_channels, modality=self.modality,
                                    output_dim=output_dim, kernel_size=self.cnn1d_kernel_size, attention=self.cnn1d_attention,
                                    dropout=self.cnn1d_dropout, root_dir=self.model_load_path)
                    
                    model.init()
            else:
                model = my_2d1d(backbone_state_dict=self.backbone_state_dict, backbone_mode=self.backbone_mode,
                                embedding_dim=self.cnn1d_embedding_dim, channels=self.cnn1d_channels, modality=self.modality,
                                output_dim=output_dim, kernel_size=self.cnn1d_kernel_size, attention=self.cnn1d_attention,
                                dropout=self.cnn1d_dropout, root_dir=self.model_load_path)
                model.init()
            
            ## Freezing the model
            if isinstance(model, TwoStreamAuralVisualModel):
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.children():
                    p.train(False)            
        else:
            raise ValueError("Unknown base_model!")

        return model

    @staticmethod
    def get_train_emotion(emotion_tag, head):

        emotion = ["Valence", "Arousal"]

        if emotion_tag == "arousal":
            if head == "sh":
                emotion = ["Arousal"]
        elif emotion_tag == "valence":
            if head == "sh":
                emotion = ["Valence"]

        return emotion

    @staticmethod
    def init_random_seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init_device(self):
        device = detect_device()
        
        if not self.args.high_performance_cluster:
            select_gpu(self.gpu)
            set_cpu_thread(self.cpu)

        print("device : " , device)

        return device
