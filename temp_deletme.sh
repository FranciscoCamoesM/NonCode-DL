# python generate_processed_dataset.py --enh E25E10 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E25E102 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P3B3 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E24A3 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E18C2 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P1B925 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P1E6 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P2F4 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P3G3 --coverage 3 --epochs 100 --lr 0.0001
# # python generate_processed_dataset.py --enh E23G7 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E23H5 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P1A3 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P1C1 --coverage 3 --epochs 100 --lr 0.0001
# python generate_processed_dataset.py --enh E22P1D10 --coverage 3 --epochs 100 --lr 0.0001


# for j in 1 #0 3 1
# do
    # for enh in E25E10 E22P3B3 E24A3 E18C2 E22P1B925 E22P1E6 E22P2F4 E22P3G3 E23G7 E23H5 E22P1A3 E22P1C1 E22P1D10 E22P1H9 E22P3B3R2 E22P3F2 E24C3 E25B2 E25B3 E25E102 E22P3B3R3 E22P1F10 E22P1E6 E22P3B4 E23G1 E22P1G5 E25E10R3
#     for enh in E22P1C1 E22P1D10 E22P1H9 E22P3B3R2 E22P3F2 E24C3 E25B2 E25B3 E25E102 E22P3B3R3 E22P1F10 E22P1E6 E22P3B4 E23G1 E22P1G5 E22P1A3 E25E10R3
#     do
#         for lr in  2
#         do
#             for wd in 1
#             do
#                 # python generate_processed_dataset.py --enh $enh --coverage $j --epochs $i --wd 1e-4 --sample_type downsample --jiggle 4
#                 # python generate_processed_dataset.py --enh $enh --coverage $j --epochs $i --wd 1e-4  --jiggle 4 # --sample_type downsample
#                 python 1_train_net.py --enh $enh --wd 1e-3 --epochs 1000 --coverage $j --jiggle 3 --seed 42 --bs 256 --lr 2e-3
#                 python 1_train_net.py --enh $enh --wd 1e-3 --epochs 1000 --coverage $j --jiggle 3 --seed 42 --bs 256 --lr 2e-5
#                 python 1_train_net.py --enh $enh --wd 1e-5 --epochs 1000 --coverage $j --jiggle 3 --seed 42 --bs 256 --lr 2e-2
#                 python 1_train_net.py --enh $enh --wd 1e-5 --epochs 1000 --coverage $j --jiggle 3 --seed 42 --bs 256 --lr 2e-6
#             done
#         done
#     done
# done



# for i in 155160463 154190360 154171846 154153348 154162385 154170127 154170637
# do
#     python wrapper_saturation.py --id $i
# done

#  python generate_processed_dataset.py --enh E25E102 --coverage 1 --epochs 100 --lr 0.0001 --wd 0 --batch_size 64

# wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/xml/ClinVarVCVRelease_2025-05.xml.gz variants_data/ClinVarVCVRelease_2025-05.xml.gz
# gunzip variants_data/ClinVarVCVRelease_2025-05.xml.gz

# for enh in E25E10 E22P3B3 E24A3 E18C2 E22P1B925 E22P2F4 E22P3G3 E22P3G3 E23H5 E22P1A3 E22P1C1 E22P1D10 E22P3F2 E24C3  E25E102 E22P3B3R3 E22P1F10 E25E10R3 E22P1E6 E22P3B4
# do
#     for coverage in 5 10 18 25
#     do
#         for beta in 0.05 0.1 0.18 0.25 0.475
#         do
#             python interpolating_labels.py --enh $enh --cov $coverage --beta $beta
#         done        
#     done
# done


# for enh in E25E10 E22P3B3 E24A3 E18C2 E22P1B925 E22P2F4 E22P3G3 E22P3G3 E23H5 E22P1A3 E22P1C1 E22P1D10 E22P3F2 E24C3  E25E102 E22P3B3R3 E22P1F10 E25E10R3 E22P1E6 E22P3B4
# for enh in E23H5 E22P1C1 E22P1B925 E22P2F4 E22P3B3 E22P1D10 E24C3 E22P3B4 E25E10 E22P1A3 E22P3B3R3 E22P1F10 E25E102 E25E10R3
# for enh in E23H5 E22P1C1 E22P1B925 E22P2F4 E22P3B3 E22P1D10 E24C3 E22P3B4 E25E10 E22P1A3 E22P3B3R3  
# for enh in E23H5 E22P1C1 
# do
#     for coverage in 5
#     do
#         for lr in 1e-3 1e-4 1e-5 1e-6 1e-7
#         do
#             for wd in 1e-2 1e-4 1e-6 1e-8
#             do
#                 python interpolating_multi_labels.py --enh $enh --coverage $coverage --lr $lr --wd $wd
#             done
#         done
#     done
# done


# for id in 181225653 193164951 182183523 192153143 188213332 194054558 189202630 205105342 191161116 191212610 195115650 192154040 205132638 194102032 188141249 191162606 191195004 191163547 182012424 187134752 191164107 191165846 205152218
# do
#     # python 3_sat_mut.py --id $id --o explainability_final --save_dir saved_models_final
#     python 3_run_deeplift.py --id $id --o explainability_final --save_dir saved_models_final --device cpu --n_refs 500
# done






# python interpolate_deeplift.py --model_dir regression_models --device cpu --n_refs 500 --enh E22P1D10 --id 2025-07-19_15-27-12 --images_save_dir explicability_final_reg/deeplift
# python interpolate_deeplift.py --model_dir regression_models --device cpu --n_refs 500 --enh E25E10 --id 2025-07-19_03-10-32 --images_save_dir explicability_final_reg/deeplift
# python interpolate_deeplift.py --model_dir regression_models --device cpu --n_refs 500 --enh E25E10R3 --id 2025-07-18_17-22-54 --images_save_dir explicability_final_reg/deeplift
# python interpolate_deeplift.py --model_dir regression_models --device cpu --n_refs 500 --enh E23H5 --id 2025-07-20_09-56-28 --images_save_dir explicability_final_reg/deeplift
# python interpolate_deeplift.py --model_dir regression_models --device cpu --n_refs 500 --enh E22P1A3 --id 2025-07-19_05-26-53 --images_save_dir explicability_final_reg/deeplift
# python interpolate_deeplift.py --model_dir regression_models --device cpu --n_refs 500 --enh E22P1F10 --id 2025-07-18_12-55-41 --images_save_dir explicability_final_reg/deeplift
# python interpolate_deeplift.py --model_dir regression_models --device cpu --n_refs 500 --enh E25E102 --id 2025-07-18_15-17-32 --images_save_dir explicability_final_reg/deeplift

python interpolate_sat_mut.py --model_dir regression_models --enh E22P1D10 --id 2025-07-19_15-27-12 --images_save_dir explicability_final_reg/satmut
python interpolate_sat_mut.py --model_dir regression_models --enh E25E10 --id 2025-07-19_03-10-32 --images_save_dir explicability_final_reg/satmut
python interpolate_sat_mut.py --model_dir regression_models --enh E25E10R3 --id 2025-07-18_17-22-54 --images_save_dir explicability_final_reg/satmut
python interpolate_sat_mut.py --model_dir regression_models --enh E23H5 --id 2025-07-20_09-56-28 --images_save_dir explicability_final_reg/satmut
python interpolate_sat_mut.py --model_dir regression_models --enh E22P1A3 --id 2025-07-19_05-26-53 --images_save_dir explicability_final_reg/satmut
python interpolate_sat_mut.py --model_dir regression_models --enh E22P1F10 --id 2025-07-18_12-55-41 --images_save_dir explicability_final_reg/satmut
python interpolate_sat_mut.py --model_dir regression_models --enh E25E102 --id 2025-07-18_15-17-32 --images_save_dir explicability_final_reg/satmut
