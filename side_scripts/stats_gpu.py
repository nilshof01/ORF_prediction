import subprocess
import pandas as pd
from io import StringIO

output = subprocess.check_output("nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv", shell=True, universal_newlines=True)

# Read the output into a pandas DataFrame
gpu_info = pd.read_csv(StringIO(output))

# Display the DataFrame
print(gpu_info)


