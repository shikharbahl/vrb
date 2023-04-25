# VRB: Affordances from Human Videos as a Versatile Representation for Robotics

**[Carnegie Mellon University](https://www.ri.cmu.edu/), [Meta AI Research](https://ai.facebook.com/research/)**

[Shikhar Bahl*](https://shikharbahl.github.io/), [Russell Mendonca*](https://russellmendonca.github.io/), [Lili Chen](http://www.lilichen.me/), [Unnat Jain](https://unnat.github.io/), [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)

[[`Paper`](https://arxiv.org/abs/2304.08488)] [[`Project`](https://robo-affordances.github.io/)] [[`Demo`](#running-vrb)] [[`Video`](https://youtu.be/Bik4s57iPsY)] [[`Dataset`]()] [[`BibTeX`](#citing-vrb)] 


![Demo of my project](./assets/rel_initial_1.gif)

Given a scene, our model (**VRB**) learns actionable representations for robot learning. VRB predicts contact points and a post-contact trajectory learned from human videos. We aim to seamlessly integrate VRB with robotic manipulation, across 4 real world environments, over 10 different tasks, and 2 robotic platforms operating in the wild.


<p float="left">
  <img src="assets/vrb_method.png?raw=true" width="75%" />
</p>

Our model takes a human-agnostic frame as input. The contact head outputs a contact heatmap (left) and the trajectory transformer predicts wrist waypoints (orange). This output can be directly used at inference time (with sparse 3D information, such as depth, and robot kinematics).


# Installation

This code uses `python>=3.9`, and `pytorch>=2.0`, which can be installed by running the following:   

First create the conda environment: 

```
conda env create -f environment.yml
```

Install required libraries: 

```
conda activate vrb
pip install -r requirements.txt
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

Either download the [model weights](https://drive.google.com/file/d/1nzahuDh4Wa0SXCPwpu9Z_MwkQGerGAhp/view?usp=sharing) and place in `models` folders or run: 

```
mkdir models
bash download_model.sh
```


# Running VRB

To run the model: 

```
python demo.py --image ./kitchen.jpeg --model_path ./models/model_checkpoint_1249.pth.tar
```

The output should look like the following: 


<p float="left">
  <img src="assets/out_kitchen.png?raw=true" width="50.25%" />
</p>


# Helpful pointers
- [Paper](https://arxiv.org/abs/2304.08488)
- [Project Website](https://robo-affordances.github.io/)
- [Project Video](https://youtu.be/Bik4s57iPsY)
- [Twitter Thread]()
- [Learning Manipulation from Watching Human Videos](https://human2robot.github.io/)
- [Learning Dexterous Policies from Human Videos](https://video-dex.github.io/)


# Citing VRB

If you find our model useful for your research, please cite the following:

```
@inproceedings{bahl2023affordances,
              title={Affordances from Human Videos as a Versatile Representation for Robotics},
              author={Bahl, Shikhar and Mendonca, Russell and Chen, Lili and Jain, Unnat and Pathak, Deepak},
              journal={CVPR},
              year={2023}
            }
```


# Acknowledgements
We thank Shivam Duggal, Yufei Ye and Homanga Bharadhwaj for fruitful discussions and are grateful to Shagun Uppal, Ananye Agarwal, Murtaza Dalal and Jason Zhang for comments on early drafts of this paper. We would also like to thank the authors of [HOI-Forecast](https://github.com/stevenlsw/hoi-forecast) [1], as the training code for VRB is adapted from their codebase. RM, LC, and DP are supported by NSF IIS-2024594, ONR MURI N00014-22-1-2773 and ONR N00014-22-1-2096.

[1] Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos. Shaowei Liu, Subarna Tripathi, Somdeb Majumdar, Xiaolong Wang. CVPR 2022. 
