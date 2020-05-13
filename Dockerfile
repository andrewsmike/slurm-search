FROM ohsucompbio/slurm
RUN yum install -y python3
RUN pip3 install hyperopt
WORKDIR /opt/all_automation
RUN mkdir -p /root/hyperparameters/search
