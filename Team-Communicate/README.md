# Environment requirements
python3.8
pip 23.2.1
mvenv (to create, activate, and deactivate the virtual environment)
python3.8-dev (obtained by sudo apt-get install python3.8-dev)
CUDA 10.2

## data_collection_client.py requirements
Cython
scikit-build
numpy
pandas
opencv-contrib-python
pyzed (ZED Python API, run get_python_api.py located in /usr/local/zed)

## network_relation.py requirements
matplotlib

## utils.py requirements
torch->dependency for pytorch3d
torchvision->dependency for pytorch3d
torchaudio->dependency for pytorch3d
fvcore->dependency for pytorch3d
iopath->dependency for pytorch3d
pytorch3d (pip install "git+https://github.com/facebookresearch/pytorch3d.git")
scipy


