args_string='
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.00001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.0001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.00001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.0001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.00001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.0001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.00001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.0001
--config ./configs/unet_raw.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.00001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.0001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.00001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.0001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.00001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.0001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.00001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.0001
--config ./configs/unet_wiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.00001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.0001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.0001 --wd 0.001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.00001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.0001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.5 --lr 0.001 --wd 0.001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.00001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.0001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.0001 --wd 0.001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.00001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.0001
--config ./configs/neureka_rawwiener.json --default_config ./configs/default_config_tuhsz2.json --rsz 0.2 --lr 0.001 --wd 0.001

'

# Read the multi-line string into an array
IFS=$'\n' read -r -d '' -a args <<< "$args_string"$'\n'

echo "Starting Process"
for i in "${args[@]}"
do
    #transfer each one to this format /users/sista/kkontras/Documents/Balance/main_mcrema_MSLR.py,config-./configs/CREMA_D/vit/concat_multiloss_ShuffleGradFinalEPIB_test.json,default_config-./configs/CREMA_D/default_config_cremad.json,fold-0,lr-0.00005,wd-5e-6,l-0.1,l_diffsq-0,lib-0,clip-None,num_samples-32,reg_by-dist_pred_cjsd,pow-1,cls-linear,batch_size-8,contr_coeff-0,shuffle_type-rand,contr_type-label,temperature-None
    split_i=($i)
    make_string="/users/sista/kkontras/Documents/Epilepsy_Challenge/irregulars_neureka_codebase/main.py"
    for j in "${split_i[@]}"
        do
        if [[ "$j" == *--* ]]; then
            j="${j//--/}"
            make_string+=",$j"
        else
            make_string+="-$j"
        fi
    done
    echo $make_string
done
