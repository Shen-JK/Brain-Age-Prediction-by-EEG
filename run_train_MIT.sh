export PATH="/opt/miniconda3/bin:$PATH"


train_stage=$1		#1-training  0-testing
log_dir=log/melspec_class_equalprob_average_$2/


if [ $train_stage -eq 1 ];then
	mkdir $log_dir
	mkdir $log_dir/code
	mkdir $log_dir/model
	cp ./run_train_MIT.sh $log_dir/code/
	cp ./train_MIT.py $log_dir/code/
	cp ./load_data.py $log_dir/code/
	cp ./model_use.py $log_dir/code/
	cp ./dataset/$2/train.scp $log_dir/code/
	cp ./dataset/$2/train_class.scp $log_dir/code/
	cp ./dataset/$2/valid.scp $log_dir/code/
	cp ./dataset/$2/valid_class.scp $log_dir/code/
	cp ./dataset/$2/test.scp $log_dir/code/
	cp ./dataset/$2/test_class.scp $log_dir/code/
fi

python train_MIT.py \
--lr 0.0001 \
--optimizer 'adam' \
--wd 1e-3 \
--log-interval 1 \
--is-training=$train_stage \
--log-dir ${log_dir} \
--start-epoch=0 \
--epochs=100 \
--fold=$2