# Optimization of Annealing Pattern Recognition Algorithms applied to Charged Particle Track Reconstruction

The purpose of this research was to investigate methods for optimizing the run time efficiency of the seeding step for quantum or simulated annealing pattern recognition algorithms for track reconstruction on the ATLAS experiment at CERN. Three sedding algorithms were tested using a standard laptop hardware configuration and on a supercomputing cluster. The fastest algorithm tested utilizes paralllel computing techniques as well as runtime compilation to runn 4.86-times faster than the original algorithm on the laptop and 24.56-times faster on the supercomputer. The speed-up demonstrated by this study will have an impact on future annealing pattern recognition projects as well as machine learning research applied to charged particle track reconstruction.

## 1  Introduction

  The ATLAS experiment at CERN measures the products of 13 TeV proton-proton collisions at
the Large Hadron Collider (LHC). Every 25ns the ATLAS detector readout systems captures a
snapshot ("Event") of the status of the detector. Each Event contains the measurements of ≈ 40
p-p collisions, and each collision produces ≈ 30 long-lived charged particles. These particles then
move outwardly through the detector apparatus and each layer of the detector records the position
of the particle at that instant of time, along with other physical properties. The mission of track
reconstruction is to take the data collected from the particle detectors and connect these dots
together in order to reconstruct the paths the particles took when they exited the detector. These
tracks can then later be examined for interesting physics.

  Around 2026 the High Luminosity LHC (HL-LHC) upgrade will come online enabling, among
other things, precision physics studies in the Higgs sector. The high luminosity improvements to
the LHC will generate 5-10 times more high energy collisions than seen previously, and upgrades
to the ATLAS Trigger and Data Acquisition (TDAQ) system will allow to process 5-10 times more
events per second. While this means more interesting physics, it also puts a heavier load on ATLAS’s 
existing track reconstruction software. The traditional algorithms for track reconstruction
are estimated to scale at greater than O(N2) run time as the number of collisions, N, increases.
For this reason, along with technological and funding constraints, experts estimate that there will
be a shortage of computational resources by a factor of three to ten times 1. So it is important to
search for new, faster algorithms to solve the track reconstruction problem.

  The difficulty of the problem encourages us to explore unconventional approaches. It has
been shown that quantum computers can more efficiently identify patterns in large sets of data.2
In order to take advantage of this gain in efficiency, the track reconstruction problem must
be reformulated in terms of a pattern recognition problem. This can be done by mapping the
set of possible tracks into a Quadratic Unconstrained Binary Optimization problem. Finding the
minimum of this function can be done on a quantum computer or a classical computer, using a
simulated quantum environment. This approach was tested by the ATLAS group at Lawrence
Berkeley National Laboratory using a D-Wave computer and found that the quantum computer
was able to solve the track reconstruction problem with comparable precision to state-of-the art
tracking algorithms. 3. However, the run time of the D-Wave quantum computer was limited by
the time needed to process the data before it could be input onto the machine. This action of
preparing the data for the quantum annealer is called seeding. The goal of this research was to
optimize the runtime performance of the seeding algorithm.

## 2  Experimental Setup and Methods
### 2.1  Code Base
  Qallse is the quantum pattern recognition software which was used as the basis for the optimization study. 
The code base is open source and was developed by Lucy Linder and the ATLAS computation group at LBNL.4

### 2.2  Detector Set Up
For the purposes of the study, an open-access dataset of MC simulated Events based on the
design of a generic HL-LHC particle tracking detector was used5. The detector geometry consists
of ten concentric cylinders which constitute the pixel detector layers. Each hit belongs to one of
these ten layers. The standard coordinate system for ATLAS is shown in Figure 3. The interaction zone 
encapsulates the area in which proton collisions occur. This region is important to track reconstruction 
because we are only interested in particles which originate from this zone. The interaction zone is defined 
as a 700 mm long cylinder along the z-axis, centered on the origin. The cartesian coordinates are as follows: 
the z-axis points along the beam line, the positive x-axis points toward the center of the LHC accelerator 
ring, and the positive y-axis points upward. The cylindrical coordinates are as follows: the radial dimension, 
R, is the distance from the center of the interaction region, the azimuthal angle, phi, is the measured in the 
x-y plane starting from the positive x-axis, the polar angle, theta, is measured in the x-z plane starting 
from the positive z-axis.

### 2.3  Overview of Seeding Algorithm Design
  The goal of seeding is to take the set of particle hits in the detector and construct a list of all
possible doublets. A doublet is a pair of consecutive hits belonging to the same particle. The algorithm can 
be broken down into four main parts:
  1. Select an inner hit, the first hit in the doublet, from the set of all hits in the detector.
  2. Using the position of this inner hit, calculate a region of interest in which the outer hit could exist,
     the second hit in the doublet.
  3. Move through the rest of the hits in the detector and check if they fall within our calculated region. If a hit
     falls within this region, then check if the inner/outer hit pair satisfies a list of physical requirements. Then,
     if the inner/outer hit pair satisfies these requirements then it should be added to the list of possible doublets.
  4. Repeat from step one until all hits in the detector have been used as inner hits.
At the end of this process, the algorithm has successfully generated a list of doublets that satisfy our requirements.

#### 2.3.1  Calculating the Region of Interest
  Given the inner hit to a doublet, the seeding algorithm must calculate a three-dimensional "Region of Interest" within
the detector where possible outer hit candidates could be contained. First, the outer hits must be contained in layers 
that are between 10 mm and 300 mm away from the inner hit. Second, the phi direction is sliced into 53 discrete slices 
and outer hits must be contained in the same or adjacent phi slices as the inner hit. Third, the line which is formed by 
the inner and outer hit pairs must intersect with the region of interest. This requirement is used to limit the
range in which outer hits can be placed in the z-direction. A diagram of the region of interest is shown in Figures 4 and 
5. The equation for calculating the z range is

--> MATH <--

Once the region of interest for possible outer hit candidates has been calculated, the algorithm can then use these ranges
to more efficiently filter through the outer hits.

#### 2.3.2  Filtering Outer Hits
  The algorithm now iterates through the set of possible outer hits and applies a series of filters in order to determine
if the inner/outer hit pairs form physically viable doublets. If an outer hit passes all the filters, then the inner/outer hit pair is added to the list of possible doublets. The following filters are applied as follows:
   * **Filter Layers**: this function returns true if the outer hit belongs to a layer in the region of interest.
   * **Filter Phi**: this function returns true if the outer hit belongs to a phi slice in the region of interest.
   * **Filter Z**: this function returns true if the outer hit is in the z range for the region of interest.
   * **Filter Doublet Length**: this function returns true if the distance between the inner and outer hits is between
     10 mm and 300 mm.
   * **Filter Horizontal Doublets**: this function returns true if the change in the theta direction is less than the maximum
     allowed change in the theta direction for doublets.
This process is continued until all hits in the data set have been used as inner hits.

### 3  Results


