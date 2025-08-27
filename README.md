# CS 470: Computer Vision and Image Processing
***Fall 2025***  
***Author: Your Name Here***  
***Original Author: Dr. Michael J. Reale***  
***SUNY Polytechnic Institute*** 

## Overview
You will need to do the following:
- Install Miniconda
  - Smaller version of Anaconda, which is a distribution of Python.  
- Create Miniconda Python environment CV
  - This will contain the python executable and all of the packages you need.
- Install Visual Code
  - Strongely recommended IDE
  
This can be done either on your own machine OR in a portable installation.

## Miniconda Installation

### Windows

Download the latest installer for Windows [here](https://docs.conda.io/projects/miniconda/en/latest/).

#### Local Machine
* Run the installer for "All Users"
* Set the destination folder to somewhere easily accessible: e.g., ```C:/miniconda3```.
* Open "Anaconda Prompt (miniconda3)" **as an administrator**; you should see ```(base)``` to the left of your terminal prompt.

#### Portable Installation
* Run the installer for "All Users"
* Set the destination folder to a location on a USB drive; for example: ```K:\Software\Windows\miniconda3```
* Uncheck "Create shortcuts"
* Create a file called ```Anaconda.bat``` in the parent directory (in my case ```K:\Software\Windows```) with the following contents:
```
@echo off
cd /d "%~dp0"
set CONDA_PATH=%CD%\miniconda3
%WINDIR%\System32\cmd.exe "/K" %CONDA_PATH%\Scripts\activate.bat %CONDA_PATH%
```
* Run this script **as an administrator**; you should see ```(base)``` to the left of your terminal prompt.

### Linux
* Download the latest version:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
* Install it into your home directory (default options are fine):
```
~/miniconda3/bin/conda init bash
```
* Close and reopen the terminal; you should see ```(base)``` to the left of your terminal prompt.

### Mac
* Open up a terminal.
* If you are on an **Intel Mac**:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
```
* If you are on an **M1 Mac**:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
sh Miniconda3-latest-MacOSX-arm64.sh
```
* Close and reopen the terminal; you should see ```(base)``` to the left of your terminal prompt.

## Environment Creation

Create your CV environment with Python 3.11:
```
conda create -n CV python=3.11
```

Before installing any packages to the new environment, activate it:
```
conda activate CV
```

```(CV)``` should now be to the left of your terminal prompt.

### Installing PyTorch
With ```CV``` activated, you will need to install PyTorch for deep learning.  

The latest instructions for installing PyTorch may be found [here](https://pytorch.org/get-started/locally/).

#### CUDA-Enabled PyTorch
If you have an NVIDIA graphics card (or intend to run this environment on a machine with one), you will want to install the CUDA-enabled version of PyTorch.

If necessary, change the version of CUDA appropriately.

##### Windows (CUDA 12.6)

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

##### Linux (CUDA 12.6)

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

##### CPU/Mac PyTorch
If you do NOT have an NVIDIA card (and/or are on a Mac), you will need to settle for the default installation:

```
pip3 install torch torchvision
```

### Verifying PyTorch
To verify Pytorch works correctly:
```
python -c "import torch; x = torch.rand(5, 3); print(x); print(torch.cuda.is_available())"
```
You should see an array printed.  

If you have an NVIDIA card and installed the CUDA version of PyTorch, you should also see the word ```True``` (otherwise, ```False``` is expected).

### Installing Other Python Packages

**Windows:** Open the prompt **as an administrator**.  

**Linux:** Make sure the commands that follow are run using ``sudo``.

Activate your environment:
```
conda activate CV
```

Then, run the following commands to install the necessary packages for this course:
```
pip3 install --upgrade diffusers["torch"] accelerate transformers
pip3 install --upgrade datasets
pip3 install opencv-contrib-python 
pip3 install scikit-learn scikit-image 
pip3 install matplotlib 
pip3 install peft
pip3 install bitsandbytes
pip3 install clean-fid
pip3 install ftfy
pip3 install av
pip3 install pytest
pip3 install jupyter
pip3 install gradio 
```

### Troubleshooting

* If you need to remove the CV environment (the LOCAL version, not the portable one):
```
conda remove --name CV --all
```

* If you encounter path issues (where conda isn't found once you activate an environment), open an Anaconda prompt as admin and try the following to add conda to your path globally: 
```
conda init
```

## Visual Code

### Local Machine
* [Download](https://code.visualstudio.com/) and install Visual Code
  * I suggest enabling the context menu options.
  * ***For Mac:*** Follow [these instructions](https://code.visualstudio.com/docs/setup/mac)

### Portable Installation
* Go [here](https://code.visualstudio.com/Download) and download the **zip version** of Visual Code for your platform (most likely x64).
* Unpack it to your USB drive.  Inside the folder for Visual Code, create a ```data``` folder.
  * This will cause Visual Code to store extensions and settings locally.

To run Visual Code:
* ***Windows:*** double-click on ```Code.exe``` in the installation folder
* ***Linux/Mac:*** navigate to the folder and run ```./code```

### Extensions
Open Visual Code and install the following extensions:
  * **Python** by Microsoft    
  * **IntelliCode** by Microsoft  
  * **Python Indent** by Kevin Rose  
  * **Git Graph** by mhutchie
  * **Markdown All in One** by Yu Zhang

### General Usage and Setup

#### Terminals
A terminal can always be created/opened with ```View menu -> Terminal```.  However, if you need to restart, click the garbage can icon on the terminal window to destroy it.

***Windows:*** Change your default terminal from Powershell to Command Prompt:
1. ```View menu -> Command Palette -> "Terminal: Select Default Profile"```
2. Choose ```"Command Prompt"```
3. Close any existing terminals in Visual Code

#### Python Interpreter
Once you open a project, to make sure you are using the correct Python interpreter:
1. Close any open terminals with the garbage can icon
2. Open a .py file
3. View -> Command Palette -> "Python: Select interpreter"
4. Choose the one associated with your ```CV``` environment (whether local or on your USB drive).

#### OpenCV + Intellisense  
* If Intellisense is highlighting OpenCV (cv2) commands red:
    1. Open ```File menu -> Preferences -> Settings```
    2. Select the ```User``` settings tab
    3. Search for ```pylint```
    4. Under ```Pylint:Args```, add an item: ```--generate-members```




