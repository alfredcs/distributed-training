#!/bin/bash
/opt/conda/bin/pip install -U --no-cache-dir mpi4py datasets transformers==4.17.0 sagemaker-training

<<comment
# --- 2 nodes, single GPU per node ---
mpirun --host algo-1:1,algo-2:1 -np 2 --allow-run-as-root --tag-output --oversubscribe \
 -mca btl_tcp_if_include eth0 -mca oob_tcp_if_include eth0 -mca plm_rsh_no_tree_spawn 1 \
 -mca pml ob1 -mca btl ^openib -mca orte_abort_on_non_zero_status 1 -mca btl_vader_single_copy_mechanism none \
 -mca plm_rsh_num_concurrent 2 -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
 -x SMDATAPARALLEL_USE_HOMOGENEOUS=1 -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 \
 -x LD_PRELOAD=/opt/conda/lib/python3.8/site-packages/gethostname.cpython-38-x86_64-linux-gnu.so \
 -verbose -x NCCL_DEBUG=VERSION -x SMDATAPARALLEL_SERVER_ADDR=algo-1 -x SMDATAPARALLEL_SERVER_PORT=7592 \
 -x SAGEMAKER_INSTANCE_TYPE=ml.g4dn.2xlarge smddprun /opt/conda/bin/python3.8 -m mpi4py \
 run_qa.py $@
comment

# --- 1 nodes, 4 GPU per node ---
mpirun --host algo-1:4 -np 4 --allow-run-as-root --tag-output --oversubscribe \
 -mca btl_tcp_if_include eth0 -mca oob_tcp_if_include eth0 -mca plm_rsh_no_tree_spawn 1 \
 -mca pml ob1 -mca btl ^openib -mca orte_abort_on_non_zero_status 1 -mca btl_vader_single_copy_mechanism none \
 -mca plm_rsh_num_concurrent 2 -x NCCL_SOCKET_IFNAME=eth0 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
 -x SMDATAPARALLEL_USE_HOMOGENEOUS=1 -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 \
 -x LD_PRELOAD=/opt/conda/lib/python3.8/site-packages/gethostname.cpython-38-x86_64-linux-gnu.so \
 -verbose -x NCCL_DEBUG=VERSION -x SMDATAPARALLEL_SERVER_ADDR=algo-1 -x SMDATAPARALLEL_SERVER_PORT=7592 \
 -x SAGEMAKER_INSTANCE_TYPE=ml.g4dn.12xlarge smddprun /opt/conda/bin/python3.8 -m mpi4py \
 run_qa.py $@

#/opt/conda/bin/python3.8 run_qa.py $@