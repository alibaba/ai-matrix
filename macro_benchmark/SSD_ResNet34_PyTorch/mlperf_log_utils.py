import collections
import os
import subprocess

import mlperf_compliance


def mlperf_submission_log(benchmark):

    framework = "PyTorch NVIDIA Release {}".format(os.environ["NVIDIA_PYTORCH_VERSION"]);

    def query(command):
        result = subprocess.check_output(
            command, shell=True, encoding='utf-8',
            stderr=subprocess.DEVNULL
            ).strip()
        return result

    def get_sys_storage_type():
        dev = query('lsblk -e 11 -ndio KNAME | head -1')
        if dev.startswith('sd'):
            transport = 'SATA'
        elif dev.startswith('hd'):
            transport = 'IDE'
        elif dev.startswith('nvme'):
            transport = 'NVMe'
        else:
            transport = '<unknown bus>'

        # FIXME: there is no way to correctly detect disk type on DGX-1, assume SSD.
        disk_type = 'SSD'
        sys_storage_type = f'{transport} {disk_type}'
        return sys_storage_type

    def get_interconnect():
        dev = query('ibstat -l | head -1')
        link_layer = query(f'ibstatus {dev} | grep "link_layer" | cut -f 2- -d" "')
        rate = query(f'ibstatus {dev} | grep "rate" | cut -f 2- -d" "')
        interconnect = f'{link_layer} {rate}'
        return interconnect

    def get_sys_mem_size():
        sys_mem_size = query(
            "grep 'MemTotal' '/proc/meminfo' | awk '{ print $2 }'"
            )
        sys_mem_size = f'{int(sys_mem_size) // (1024 * 1024)} GB'
        return sys_mem_size

    def get_sys_storage_size():
        sizes = query(
            'lsblk -e 11 -dno SIZE | sed \'s/ //g\''
            ).split()
        sizes_counter = collections.Counter(sizes)
        sys_storage_size = ' + '.join([f'{val}x {key}' for key, val in sizes_counter.items()])
        return sys_storage_size

    def get_cpu_interconnect(cpu_model):
        if cpu_model == '85':
            # Skylake-X
            cpu_interconnect = 'UPI'
        else:
            cpu_interconnect = 'QPI'
        return cpu_interconnect

    gcc_version = query(
        'gcc --version |head -n1'
        )

    os_version = query(
        'cat /etc/lsb-release |grep DISTRIB_RELEASE |cut -f 2 -d "="',
        )
    os_name = query(
        'cat /etc/lsb-release |grep DISTRIB_ID |cut -f 2 -d "="',
        )

    cpu_model = query(
        'lscpu |grep "Model:"|cut -f2 -d:'
        )
    cpu_model_name = query(
        'lscpu |grep "Model name:"|cut -f2 -d:'
        )
    cpu_numa_nodes = query(
        'lscpu |grep "NUMA node(s):"|cut -f2 -d:'
        )
    cpu_cores_per_socket = query(
        'lscpu |grep "Core(s) per socket:"|cut -f2 -d:'
        )
    cpu_threads_per_core = query(
        'lscpu |grep "Thread(s) per core:"|cut -f2 -d:'
        )

    gpu_model_name = query(
        'nvidia-smi -i 0 --query-gpu=name --format=csv,noheader,nounits'
        )
    gpu_count = query(
        'nvidia-smi -i 0 --query-gpu=count --format=csv,noheader,nounits'
        )

    sys_storage_size = get_sys_storage_size()

    hardware = query(
        'cat /sys/devices/virtual/dmi/id/product_name'
        )

    network_card = query(
        'lspci | grep Infiniband | grep Mellanox | cut -f 4- -d" " | sort -u'
        )
    num_network_cards = query(
        'lspci | grep Infiniband | grep Mellanox | wc -l'
        )
    mofed_version = query(
        'cat /sys/module/mlx5_core/version'
        )
    interconnect = get_interconnect()

    cpu = f'{cpu_numa_nodes}x {cpu_model_name}'
    num_cores = f'{int(cpu_numa_nodes) * int(cpu_cores_per_socket)}'
    num_vcores = f'{int(num_cores) * int(cpu_threads_per_core)}'
    cpu_interconnect = get_cpu_interconnect(cpu_model)

    sys_storage_type = get_sys_storage_type()
    sys_mem_size = get_sys_mem_size()

    num_nodes = os.environ.get('SLURM_NNODES', 1)

    nodes = {
        'num_nodes': num_nodes,
        'cpu': cpu,
        'num_cores': num_cores,
        'num_vcpus': num_vcores,
        'accelerator': gpu_model_name,
        'num_accelerators': gpu_count,
        'sys_mem_size': sys_mem_size,
        'sys_storage_type': sys_storage_type,
        'sys_storage_size': sys_storage_size,
        'cpu_accel_interconnect': cpu_interconnect,
        'network_card': network_card,
        'num_network_cards': num_network_cards,
        'notes': '',
        }

    libraries = {
        'container_base': f'{os_name}-{os_version}',
        'openmpi_version': os.environ['OPENMPI_VERSION'],
        'mofed_version': mofed_version,
        'cuda_version': os.environ['CUDA_VERSION'],
        'cuda_driver_version': os.environ['CUDA_DRIVER_VERSION'],
        'nccl_version': os.environ['NCCL_VERSION'],
        'cudnn_version': os.environ['CUDNN_VERSION'],
        'cublas_version': os.environ['CUBLAS_VERSION'],
        'trt_version': os.environ['TRT_VERSION'],
        'dali_version': os.environ['DALI_VERSION'],
        }

    entry = {
        'hardware': hardware,
        'framework': framework,
        'power': 'N/A',
        'notes': 'N/A',
        'interconnect': interconnect,
        'os': os.environ.get('MLPERF_HOST_OS', ''),
        'libraries': str(libraries),
        'compilers': gcc_version,
        'nodes': str(nodes),
        }

    mlperf_compliance.mlperf_log.setdefault(
        root_dir=os.path.dirname(os.path.abspath(__file__)),
        benchmark=benchmark,
        stack_offset=0,
        extra_print=False
        )

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_BENCHMARK,
        value=benchmark,
        )

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_ORG,
        value='NVIDIA')

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_DIVISION,
        value='closed')

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_STATUS,
        value='onprem')

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_PLATFORM,
        value=f'{num_nodes}x{hardware}')

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_ENTRY,
        value=str(entry))

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_POC_NAME,
        value='Paulius Micikevicius')

    mlperf_compliance.mlperf_log.mlperf_print(
        key=mlperf_compliance.constants.SUBMISSION_POC_EMAIL,
        value='pauliusm@nvidia.com')
