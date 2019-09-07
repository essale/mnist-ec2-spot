#!/bin/bash
EC2_INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
EC2_AVAIL_ZONE=`curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone`
EC2_REGION="`echo \"$EC2_AVAIL_ZONE\" | sed -e 's:\([0-9][0-9]*\)[a-z]*\$:\\1:'`"
DEVICE_NAME="/dev/xvdb"
DATA_STATE="unknown"
until [ "$DATA_STATE" == "attached" ]; do
        DATA_STATE=$(aws ec2 describe-volumes \
                --region $EC2_REGION \
                --filters \
                Name=attachment.instance-id,Values=$EC2_INSTANCE_ID \
                Name=attachment.device,Values=$DEVICE_NAME \
                --query Volumes[].Attachments[].State \
                --output text)
        echo $DATA_STATE
        sleep 5
done
echo "volume is ready"

echo "Start mounting volumes"
sudo mkdir -p /dl
sudo mkfs -t xfs /dev/xvdb
sudo mount /dev/xvdb /dl
sudo chown -R ubuntu: /dl/
cd /dl
mkdir -p datasets
mkdir -p checkpoints

[ "$(ls -A /dltraining/datasets/)" ] && echo "Not Empty" || curl -o /dl/datasets/mnist.npz https://s3.amazonaws.com/img-datasets/mnist.npz

sudo -H -u ubuntu bash -c "source /home/ubuntu/anaconda3/bin/activate tensorflow_p27; python ec2_spot_keras_training.py "
