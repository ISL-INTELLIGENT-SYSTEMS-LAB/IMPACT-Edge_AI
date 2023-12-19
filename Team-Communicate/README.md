--- environment requirements (do this first) ---
python3.8
pip 23.2.1
mvenv (to create, activate, and deactivate the virtual environment)
python3.8-dev (obtained by sudo apt-get install python3.8-dev)
CUDA 10.2
--- for data_collection.py ---
Cython
scikit-build
numpy
pandas
opencv-contrib-python
pyzed (ZED Python API, run get_python_api.py located in /usr/local/zed)
--- for network_relation.py ---
matplotlib
--- for utils.py ---
torch->dependency for pytorch3d
torchvision->dependency for pytorch3d
torchaudio->dependency for pytorch3d
fvcore->dependency for pytorch3d
iopath->dependency for pytorch3d
pytorch3d (pip install "git+https://github.com/facebookresearch/pytorch3d.git")
scipy


