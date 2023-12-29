config_preprocessing = {
    # "raw_video_path": r"H:\Affwild2\raw_video",
    # "test_video_path": r"H:\Affwild2\Test_Set",
    # "annotation_path": r"H:\Affwild2\annotations\VA_Set",
    # "image_path": r"H:\Affwild2\cropped_aligned",
    # "output_path": r"H:\Affwild2_processed",
    "raw_video_path": r"../data/Affwild2/train_video/batch1",
    "test_video_path": r"../data/Affwild2/test_video/new_vids",
    "annotation_path": r"../data/Affwild2/annotations/VA_Set",
    "image_path": r"../data/Affwild2/cropped_aligned_images1/cropped_aligned",
    "output_path": r"output",

    # "aural_feature_list": ["mfcc", "egemaps", "vggish"],
    # "opensmile_exe_path": r"D:\opensmile-3.0-win-x64\bin\SMILExtract.exe",
    # "opensmile_config_path": r"D:\opensmile-3.0-win-x64\config",
    "aural_feature_list": ["mfcc", "egemaps", "vggish"],
    "opensmile_exe_path": r"ABAW2/opensmile-3.0-linux-x64/bin/SMILExtract",
    "opensmile_config_path": r"ABAW2/opensmile-3.0-linux-x64/config",

    # "openface_config": {
    #         "openface_directory": "D:\\OpenFace-master\\x64\\Release\\FeatureExtraction",
    #         "input_flag": " -f  ",
    #         "output_features": " -2Dfp ",
    #         "output_action_unit": " -aus ",
    #         "output_image_flag": " -simalign ",
    #         "output_image_format": " -format_aligned jpg ",
    #         "output_image_size": " -simsize 48",
    #         "output_image_mask_flag": " -nomask ",
    #         "output_filename_flag": " -of ",
    #         "output_directory_flag": " -out_dir "
    #     }
    "openface_config": {
            "openface_directory": "ABAW2/OpenFace-OpenFace_2.2.0/exe/FeatureExtraction",
            "input_flag": " -f  ",
            "output_features": " -2Dfp ",
            "output_action_unit": " -aus ",
            "output_image_flag": " -simalign ",
            "output_image_format": " -format_aligned jpg ",
            "output_image_size": " -simsize 48",
            "output_image_mask_flag": " -nomask ",
            "output_filename_flag": " -of ",
            "output_directory_flag": " -out_dir "
        }
}

config_processing = {

}


# I extracted vggish feature using remote server.
# config_preprocessing = {
#     "raw_video_path": r"/home/zhangsu/affwild/raw_video",
#     "annotation_path": r"/home/zhangsu/affwild/annotations/VA_Set",
#     "image_path": r"H:\Affwild2\cropped_aligned",
#     "output_path": r"/home/zhangsu/affwild_processed",
#
#     "aural_feature_list": ["mfcc", "vggish"],
#     "opensmile_exe_path": r"D:\opensmile-3.0-win-x64\bin\SMILExtract.exe",
#     "opensmile_config_path": r"D:\opensmile-3.0-win-x64\config",
#
# }
