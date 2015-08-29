# train neural network
#./train.sh --file=/tmp/train_data.csv --output_nodes=1 --holdback=0.1 --k=8 --min_weight=-0.1 --max_weight=0.1 --learning_rate=0.2 --momentum=0.3 --epochs=20000 --model_output=/tmp --hidden_nodes=16,8

./train.sh --file=/tmp/train_data.csv --output_nodes=1 --holdback=0.2 --k=8 --min_weight=-0.1 --max_weight=0.1 --learning_rate=0.25 --momentum=0.4 --epochs=5000 --model_output=/tmp --hidden_nodes=48

# single model results
# neural network light run on training and then validation data
./run.sh /tmp/weights.bin /tmp/train_data.csv |grep "out" |sed 's/^.*net = \[ //; s/ ].*$//' >/tmp/train_out.csv
./run.sh /tmp/weights.bin /tmp/valid_data.csv |grep "out" |sed 's/^.*net = \[ //; s/ ].*$//' >/tmp/valid_out.csv
./run.sh /tmp/weights.bin /tmp/test_data.csv |grep "out" |sed 's/^.*net = \[ //; s/ ].*$//' >/tmp/test_out.csv

# k-fold results
for((i=1;i<=32;i++)); do
  echo "model" $i
  ./run.sh /tmp/weights_$(printf "%02d" $i).bin /tmp/train_data.csv |grep "out" |sed 's/^.*net = \[ //; s/ ].*$//' >/tmp/train_out_$(printf "%02d" $i).csv
  ./run.sh /tmp/weights_$(printf "%02d" $i).bin /tmp/valid_data.csv |grep "out" |sed 's/^.*net = \[ //; s/ ].*$//' >/tmp/valid_out_$(printf "%02d" $i).csv
  ./run.sh /tmp/weights_$(printf "%02d" $i).bin /tmp/test_data.csv |grep "out" |sed 's/^.*net = \[ //; s/ ].*$//' >/tmp/test_out_$(printf "%02d" $i).csv
done



## do a search for optimum # of hidden nodes (1 layer) from 1 to 48 (ncols^2).
## report to a csv (search.csv): do k-folding 10 times for each level.
## num_nodes, best_test_epoch, best_test_rmse
for((i=72;i<=120;i+=16)); do for((j=1;j<=10;j++)); do ./train.sh --file=/tmp/train_data_small.csv --output_nodes=1 --holdback=0.2 --k=8 --min_weight=-0.1 --max_weight=0.1 --learning_rate=0.25 --momentum=0.4 --epochs=500 --model_output=/tmp --hidden_nodes=$i |grep K-Fold |awk -v ii=$i '{print ii "," $9 "," $17}' ;done ;done >>/tmp/search.csv


