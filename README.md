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

## 3  Results

  Four implementations of seeding algorithms were tested. Each implementation uses different
data structures and python libraries but they follow the same outline covered in the previous
section.

### 3.1  Loop Method

  This is the original method used in the D-Wave study. The Loop Method is written in standard
python and does not use any external libraries. The method utilizes six loops to iterate over all
possible inner/outer hit combinations. The benefit of this approach is that each loop allows the
algorithm to iterate layer-wise, phi-wise, and hit-wise. This means that the function can filter
layers and phi slices without ever looking at the hits inside these regions. This cuts down on the
average number of calculations needed per outer hit.

  The algorithm was tested in a virtual machine on a laptop using two Intel Core i7 processors.
The algorithm was run on 20 samples from the simulated data set. Each sample contains an increasing 
fraction of the hits expected in HL-LHC conditions starting from 5 percent of the HL-LHC data
up to 100 percent, in 5 percent increments. A graph of the resulting runtimes are shown in Figure 6.

  Then, the algorithm was tested on a CPU node with 64 threads on the NERSC Cori Haswell
supercomputer. The algorithm was again run on twenty sample data set in 5 percent increments
starting from 5 percent of the data up to 100 percent of the data set. A graph of the resulting
runtimes is shown in Figure 7.

### 3.2  GUvectorize Method

  This method loads all the hits into a large two-dimensional Numpy array. The method also uses
the Numba python library 6. The GUvectorize function decorator from the Numba library was
used to convert the filter function from a standard python implementation into a Generalized Universal 
Numpy function. Generalized Universal Numpy functions allow for fast computation across
Numpy arrays. Furthermore, the jit function decorator was applied to the function responsible
for calculating the region of interest as well as other auxiliary functions. Finally, the nopython
flag was set to true for all functions implemented with Numba. This allows the functions to be
compiled at runtime and run as fast as code written in C.

  The same tests that were run on the loop method were run on the GUvectorize method and
are shown in Figures 6 and 7.

### 3.3 Parallel Method

  This method uses the same basic data structure as the GUvectorize method. The goal of this
method is to run the inner hit loop in parallel. This is done by using the jit decorator on the
inner hit loop and setting the parallel key word argument to true. Numba does not support calls
to universal Numpy functions inside jit functions, so jit was used to compile the filter function
instead of GUvectorize.

  The results are shown in Figures 6 and 7
  
### 3.4  Pandas Methods

  This method is analogous to the Numpy method. However, this method replaces the twodimensional Numpy array 
used in the Numpy method with a Pandas data frame. This algorithm performed worse than the Numpy method, taking 
120 seconds to run on 5 percent of the dataset. This is due to the way in which Pandas handles filtering of data 
frames. For each inner hit iteration, the data in the data frame is copied into a new data frame instance. 
The data frame contains a large amount of data and the inner hit iteration is executed for each row in the frame, so
the combined time to copy this data frame adds up to a substantial amount of runtime. A second
Pandas method was tried in an attempt to reduce this filtering problem as well as incorporate the
layer and phi-wise iteration advantage that the Loop method benefits from. This was done by
binning the z-direction such that the data frame could be multi-indexed so that the layer bin, phi
bin, and z bin were each their own dimension in the data frame. This allowed for index slicing
rather than the boolean indexing that the other methods used. However, this method still suffered
from slow downs, taking 90 seconds to run on 5 percent of the data set.

## 4  Data Analysis

  On the complete data set, the parallel method ran about 4.68-times faster than the original
loop method on a laptop and 24.56-times faster on Cori. The parallel method achieves the best
performance by running the inner hit loop in parallel. The inner hit loop is a prime candidate
for parallelization for two reasons. First, the computations per iteration are complex enough to
outweigh the latency time associated with running in parallel so gains in efficiency are possible.
Each iteration through the inner hit loop takes the laptop configuration roughly three milliseconds
to complete, on average. Second, each inner hit iteration does not depend on the others. In other
words, the doublets which are formed using one inner hit are not changed by the doublets that are
formed using a different inner hit. This allows for the loop to be run non-sequentially and achieves
significant speed ups.

  There is only a minimal difference in runtimes between the GUvectorize method and the
parallel method when run on the laptop. This is because only two cores are being accessed by the
virtual machine which limits the efficiency of parallelization. The speed ups that are being seen
when the two algorithms are run on the laptop are most likely the result of the compilation of the
code at runtime. And, the slight edge the parallel method gains over the GUvectorize method on
the laptop is likely the result of the extra core being used.

  The real difference between the two algorithm designs can be seen when they are run on
Cori. On this hardware configuration, the parallel method has access to 64 hardware threads
which maximizes the algorithm’s capabilities. Here, the parallel method out performs the GUvectorize 
method by a significant margin even though both algorithms benefit from being compiled
at runtime. This suggests the bulk of the parallel method’s speed up derives from exploiting the
additional threads.

  Finally, the loop method is the slowest of the three seeding methods because it is not compiled
at runtime and does not run in parallel. The method is limited because its large nested loop
structure can not be run non-sequentially. The outcome of one loop is dependent on the other.
This limits the algorithm to running its processes serially and does not take full advantage of the
hardware configuration it is being run on.


## 5 Conclusion

  The objective of the research was to improve the design of the seeding algorithm for annealing
pattern recognition applied to charged particle track reconstruction. The python libraries Numpy
and Numba were used to compile the code at runtime and run the computationally expensive loops
in parallel. Four algorithm methods were developed and tested on a standard laptop configuration
and on a supercomputing cluster. It was shown that the parallel seeding algorithm ran 4-times
faster than the old seeding algorithm on the laptop configuration and 24-times faster on the super
computing cluster. The future of the research will include modifying the parallel method with
CuPy so that it can be tested on an NVIDIA GPU. The results of this study will improve the
runtime of future annealing pattern recognition and machine learning studies applied to charged
particle track reconstruction.


## 6 Acknowledgements

  Thanks to Paolo Calafiura for his mentorship and guidance throughout my research. Thanks
to Rollin Thomas and Laurie Stephey for their programming support. This work was prepared
in partial fulfillment of the requirements of the Berkeley Lab Undergraduate Research (BLUR)
Program, managed by Workforce Development and Education at Berkeley Lab.








