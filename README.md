# For Jetson Nano Super
* Jetpack: 6.2 (Confirm: *apt show nvidia-jetpack*)
* CUDA: 12.6 (Confirm: *nvcc --version*)
* Python 3.10 (Confirm: *python --version*)

## Install the dependencies
* *python -m venv myenv_jetson*
* *pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 -r requirements.txt*

**Note:** I've built the torchtext from source for jetson and have provided the wheel, if the user feels uncomfortable using the built wheel, one can build it using the **torchtext_install_0_18.sh** script.