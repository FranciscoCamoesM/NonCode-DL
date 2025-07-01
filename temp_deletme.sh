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


for j in 10 3 1
do
    for enh in E25E10 E22P3B3 E24A3 E18C2 E22P1B925 E22P1E6 E22P2F4 E22P3G3 E23G7 E23H5 E22P1A3 E22P1C1 E22P1D10 E22P1H9 E22P3B3R2 E22P3F2 E24C3 E25B2 E25B3 E25E102 E22P3B3R3 E22P1F10 E25E10R3 E22P1E6 E22P3B4 E23G1 E22P1G5 
    do
        for i in 1000
        do
            for wd in 1e-2
            do
                # python generate_processed_dataset.py --enh $enh --coverage $j --epochs $i --wd 1e-4 --sample_type downsample --jiggle 4
                # python generate_processed_dataset.py --enh $enh --coverage $j --epochs $i --wd 1e-4  --jiggle 4 # --sample_type downsample
                python 1_train_net.py --enh $enh --coverage $j --epochs $i --wd $wd --jiggle 3 --seed 42 --bs 256 --lr 2e-6
            done
        done
    done
done

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
