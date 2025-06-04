# sim_a_splat

> [!WARNING]
> The repo is still under heavy development and may not work as expected. 

The repo allows one to attach a simulator to a Gaussian splat, in this case the [Drake Robotics Toolbox](https://drake.mit.edu/) to "simulate" the Gaussian splat. This can be helpful in automated data collection, especially of visually realistic camera data that is often necessary to train [diffusion models](https://diffusion-policy.cs.columbia.edu/). 

## Background
The repo relies on the Gaussian splat trained using the `nerfstudio` library. Follow the instructions [here](https://docs.nerf.studio/quickstart/custom_dataset.html). 

## Installation
The repository can be easily accessed using [pixi](https://pixi.sh/latest/), an alternative to conda and pip. 
Installation instructions can be seen [here](https://pixi.sh/latest/). 

## Usage

> [!NOTE]
> If you want to run the example, you can skip step-1. 

### Step-1
Once the Gaussian splat is trained and placed in the assets directory, run the `match_splat.py` as a notebook file to segment the elements in your Gaussian splat that are required to be connected to the simulator. In this repo an example is provided with a pre-trained environment. Upon extracting the necessary masks from the previous step, proceed to the next step.

### Step-2
Now run the demo code to visualise the example. Two browser windows can be opened beside each other, one for [viser](https://github.com/nerfstudio-project/viser), and another for the [meshcat visualiser](https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1_meshcat.html) from Drake. 

```
pixi run python demo_pusht_splat.py 
```

Upon running, you should see an interface similar to the one shown below, 

![sim_a_splat example](assets/sim_a_splat_example.gif)

The demo uses Microsoft Edge browser with a split window functionality to view the two viewers simultaneously. 

### Next steps
The "digital twin" environment is wrapped as an env compatible with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), meaning, that the Guassian splat can be used to perform training or for collecting data using the typical RL pipeline. 

### Common issues
> In some cases an older version of `viser (0.2.7)` since that is the locked version by `nerfstudio (1.1.5)` resulting in an error

This is now fixed by installing the fork of nerfstudio that ports the updated viser version.

### Acknowledgements
- The repo uses the code from [splatnav](https://github.com/chengine/splatnav) for loading and viewing the Gaussian splat.
- The communication pipeline between Drake and the Gaussian splat is inspired from the [rtxdrake](https://github.com/lvjonok/rtxdrake) project. 
- The pushT task and the teleoperation interface is implemented from the [diffusion-policy](https://github.com/real-stanford/diffusion_policy) repository. 
- Similar projects: [SplatSim](https://github.com/qureshinomaan/SplatSim) and [Splat-Sim](https://github.com/cancaries/Splat-Sim). 
