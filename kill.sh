cd $(dirname $0)
ps -aux | grep "train.py" | awk '{ print $2 }' | sudo xargs kill -9

# nohup ./run.sh &