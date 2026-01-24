mode='short' 
length=12
multi_channel_input=False
use_satellite_memory=1
gpu_id=3
model_dir='model_'$mode$multi_channel_input$use_satellite_memory
results_dir='results_'$mode$multi_channel_input$use_satellite_memory
input_len=$length
output_len=$length

if [ $mode = 'short' ]; then
    echo "the same"
    training_data_dir='/Solar_Radiance_MultiModal/solarcube_hybrid_sat_insitu_train.h5'
    training_data_points_dir='/Solar_Radiance_MultiModal/solarcube_hybrid_sat_insitu_train_points.h5'
    test_data_dir='/Solar_Radiance_MultiModal/solarcube_hybrid_sat_insitu_test.h5'
    test_data_points_dir='/Solar_Radiance_MultiModal/solarcube_hybrid_sat_insitu_test_points.h5'
else
    training_data_dir='/Solar_Radiance_MultiModal_96/solarcube_hybrid_sat_insitu_long_train.h5'
    training_data_points_dir='/Solar_Radiance_MultiModal_96/solarcube_hybrid_sat_insitu_long_train_points.h5'
    test_data_dir='/Solar_Radiance_MultiModal_96/solarcube_hybrid_sat_insitu_long_test.h5'
    test_data_points_dir='/Solar_Radiance_MultiModal_96/solarcube_hybrid_sat_insitu_long_test_points.h5'
    echo "not the same"
fi

batch_size=64
lr=2e-4

python train_sat_solar_cast.py \
--model_dir $model_dir \
--results_dir $results_dir  \
--input_len $input_len \
--output_len $output_len \
--training_data_dir $training_data_dir \
--training_data_points_dir $training_data_points_dir \
--test_data_dir $test_data_dir \
--test_data_points_dir $test_data_points_dir \
--batch_size $batch_size \
--lr $lr \
--multi_channel_input $multi_channel_input \
--gpu_id $gpu_id \
--use_satellite_memory $use_satellite_memory

python evaluate.py --data_path $results_dir'/'