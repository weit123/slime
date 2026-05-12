# Retrieve multi-node config
if [ ! -f /dockerdata/hostfile ]; then
    awk '{print $1}' /etc/taiji/hostfile > /dockerdata/hostfile
    tail -n +2 /dockerdata/hostfile > /dockerdata/hostfile_other
fi

# Configure local shell
source ~/.bashrc
export PYTHONPATH="/root/Megatron-LM/:${PYTHONPATH}"

ulimit -n 1048576 || ulimit -n 65536 || true
echo "nofile: $(ulimit -n)"

ray stop --force
sleep 2

ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8080

# Wait for head node to be fully ready
sleep 2

# Start Ray on all other nodes
for node in $(cat /dockerdata/hostfile_other | awk '{print $1}'); do
    echo "Starting Ray on node: $node"
    ssh root@$node bash -l -c "'
        source ~/.bashrc

        # Set PYTHONPATH for Megatron
        export PYTHONPATH=\"/root/Megatron-LM/:\${PYTHONPATH}\"

        # Set ulimit
        ulimit -n 1048576 || ulimit -n 65536 || true
        echo \"nofile: \$(ulimit -n)\"

        # Stop any existing Ray instance
        ray stop --force
        sleep 2

        # Start Ray worker
        ray start --address='${CHIEF_IP}:6379' && echo \"Ray worker started successfully on $node\" || echo \"Failed to start Ray worker on $node\"
    '"
done

python - <<'PY'
import ray
ray.init()
print("OK", ray.cluster_resources())
PY