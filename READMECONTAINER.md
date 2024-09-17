# SimVP
This is a conversion / extraction and to some extent a modification
of SimVP from the [OpenSTL Benchmark](https://github.com/chengtan9907/OpenSTL).
You can read more about SimVP [here](https://arxiv.org/abs/2206.05099) and
[here](https://arxiv.org/abs/2211.12509).


# Getting started
This document will be largely the same as the original README.md but uses containers to execute code
To get started we want to clone this repository, build the container, and run commands inside the container.
We will use charliecloud in this document.

## Making the container
```
$ git clone https://github.com/cculanlaiml/simvp_icmla2024.git
$ cd simvp_icmla2024
```
Now that we have the code we will build the container.

```
$ ch-image build -t simvp .
```

This command will tell the charliecloud image builder to build our container using the tag "simvp" and use the current directory as the context directory.

```
$ ch-image list
nvcr.io/nvidia/pytorch:24.08-py3
simvp
```

Behold we now have our container ready for executing code inside the container.

# Running simvp in the container
Common container tools like docker use a concept called "volumes" which enables you to share a folder from the host with the container. Charliecloud achieves this using bind mounts.

## Generate data
First we need to create a directory on the host where we plan to store out generated data.

```
cd $HOME
mkdir data_boiling
```

We will share this directory with the container so we can visualize our output later and keep our data on our host.
We use `ch-run` to execute commands in our container.

```
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/gen_initials.py --simulation Boiling --datafolder ./mnt/0 --num_initials 10 --image_height 32 --image_width 32
```

This will create 10, 32x32 initial condition images for the Boiling Water "simulation"
and place them into the directory `$HOME/data_boiling`.  Adjust as necessary. 

Note that we use a unique experiment hex ID which you will likely find obnoxious
(and long) but we find very important for tracking experiments.  Because of this
randomness, I can't tell you precisely what the directory names will be.  You 
will get something that looks a bit like this:

```
ls $HOME/data_boiling/
1e7e3e039d708148d029f601213983794d5e00fd62b962691be86cde0a7c87fe_boiling_1
36f80972ae33598c096b9743f40e40d41fa598bf21cd77e82d5176db3687d733_boiling_8
380eb6e439e5c648d6363c7f6c0b6d399b40c451fdaf816b1517fa843b6b0a54_boiling_4
43a25f7eefeae2627b409ec70c6fc74c7544cc3029dbcb32ba766b904d83a6ff_boiling_0
4e0c78cac27b881ccf56554759cf17bd814b7701afd3f7edc10c65a091e0ce12_boiling_7
64b6c84360ac588ef8512618c694cbaa118861fec2792b601504138a1ffba554_boiling_2
9c8d0141ae14132c8b76413b62ac3e112c09f68c81abdeeb5e11d322eb8984cc_boiling_5
b9d7599a826a77c87c6bd08cad37c8b22cf59ba12c1d17cb22b62a083e1e12f7_boiling_9
f138a9e060037fde03eb01ed1001a23896e1d7c6a90b39ba1c42afa8c502acf6_boiling_3
f64c70c8cba367b08cddb456eab39b9d27489a92cda1b120bf3ac786a93248a6_initials.json
f740e5c4530457fa928fc790986e00345d213631f38d913790701360dd7b93f1_boiling_6
```

If you look in one of those dirs, you'll see `0.npy` (the initial condition)
and you can open that up and look at the 32x32 "image" if you'd like.

## Running the Simulation
Please remember, these "simulations" are very *VERY* simple, they are used to
allow us to explore and create datasets of variable size very quickly.
This entire workflow does work with real modsims, so feel free to replace
this with real datasets if you wish.

```
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/gen_samples.py --datafolder ./mnt/0 --total_sample_length 100 --normalize
```

This will take all the subdirs in `$HOME/data_boiling`, pick up their initial condition,
and run the simulation for 100 time steps.  We like the `--normalize` option, but you can
play with it.  If you look at the directories now, you'll see `0.npy -> 99.npy`.

## Generating Data Loaders
OK now we'll build some data loaders for PyTorch!  Let's start with a really
small one so we can understand what we're looking at:

```
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/prep_loaders.py --datafolder /mnt/0 --num_samples 5 --sample_start_index -1 --total_length 3
```

OK let's break this down.  First of all, this will create a file such as:
`./data_boiling/015af08945eb8be3c70f5014b9cdf49e3e18dc05aa6185f5cdbcd85b2ed91c93_loaders.json`.  Let's
open it and check it out.

This is JSON structured data.  Notice we pulled in 5 samples - each of 3 length (3 consecutive numpy
files).  Notice they are random where these start (that's the `--sample_start_index -1` part).

## Running a More Realistic Workflow
OK so now that we did a simple one, let's do a "real" one - with 500 simulations:

```
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/gen_initials.py --simulation Boiling --datafolder ./mnt/0 --num_initials 500 --image_height 32 --image_width 32
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/gen_samples.py --datafolder ./mnt/0 --total_sample_length 100 --normalize
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/prep_loaders.py --datafolder ./mnt/0 --num_samples 50 --sample_start_index -1 --total_length 3
```

This'll create 50 elements the data loader, broken into train/val/test, and each of them are 3 length.  Notice
that since our original dataset had 500, 100 length samples, this will pick exactly 50 of those 500 (each pulling
only 3 images from that dataset).  The way it's currently implemented, if you pick a number > than are
in that directory, it won't pick directories it has already sampled from.

Let's say we want to do a 10 input, 10 output predictor for our ML model training later - how
might we do that?

Let's try this!  We don't have to recreate the data - just the loaders!

```
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/prep_loaders.py --datafolder ./mnt/0 --num_samples 400 --sample_start_index -1 --total_length 20
```

# Training the ML Model
Now that we have the data loaders, we can train a SimVP model on it.  Let's do it!

SimVP has a distributed and serial versions of the python programs, so for this simple
test we're going to use `train_simvp_standalone.py`.  You can see the options with `--help`.
Below, we're going to use some very basic things to get it going.

## Which Loader?
OK so let's pick the 50 samples loader we generated above.  Since we have this hash string,
I can't guarantee to you that the string below is the same on my machine as your's.  So let's
do this:

> **Note**: Make sure you replace the string below with the loader you built above on your
> machine

```
export MY_LOADER='$HOME/data_boiling/76dd1dd95b0990f1e364a48b284e531b035ed92ad96e4a61aa91ae086ac86b0b_loaders.json'
```

## Test Running Training a Model
Let's try it with some simple options:

Understand that, to run this on a mac / non-GPU / non-CUDA we have to VASTLY adjust
the model - so try not and focus on the loss and accuracy of the model, just note
that even for this crappy model it *is* learning - we can see the SSIM increase
and the loss decrease.

```
export LOCAL_RANK=0  # for now, we have to do this, it's a bug in our code
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/train_simvp_standalone.py --datafile_in $MY_LOADER --pre_seq_length 10 --aft_seq_length 10 --epoch 10 --batch_size 2 --val_batch_size 2 --device cpu --config_file /simvp/configs/SimVP_super_simple.py
```

You might get a result that looks sort of like this:

```
. . .
Epoch: 9/10, Steps: 140 | Lr: 0.0000488 | Train Loss: 0.0301789 | Vali Loss: 0.0358254
val ssim: 0.3985464087105036, mse: 36.68523406982422, mae: 129.8966064453125, rmse: 5.791803359985352, psnr: 63.43033218383789

Epoch: 10/10, Steps: 140 | Lr: 0.0000000 | Train Loss: 0.0302097 | Vali Loss: 0.0354943
val ssim: 0.40610307019547975, mse: 36.34614944458008, mae: 128.48306274414062, rmse: 5.762585639953613, psnr: 63.480430603027344

Training time: 54.0 seconds
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing Debug  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Loading model from work_dirs/Debug/simvp_model.pth
ssim: 0.3914077803882872, mse: 36.18033218383789, mae: 128.90582275390625, rmse: 5.855959415435791, psnr: 63.15140914916992
Total time: 55.0 seconds
```

## Running a Trained Model in Inference
OK now that we've trained a (presumably shitty / quick and dirty) model, let's run
it forward in inference mode and take a look at what it produces!

First, we run the model in inference mode.  It's important to understand this is using our same loader, but
recall that since when we built that loader we included a train/val/test split, this picks up the
`test` portion of that json.  So these *ARE* sets of data that were not included in train/val.

```
ch-run -b $HOME/data_boiling:/mnt/0 simvp -- python /simvp/test_simvp_standalone.py --datafile_in $MY_LOADER --pre_seq_length 10 --aft_seq_length 10 --device cpu --config_file ./configs/SimVP_super_simple.py
```

You'll see a (presumably, if you're following along) very quick (because of how simple of a model we're testing)
run of the saved model.  The SSIM should be pretty god awful (below, I get 0.39 out of a possible 1.0 / best score):

```
. . .
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing Debug  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Loading model from work_dirs/Debug/simvp_model.pth
ssim: 0.3922913267507359, mse: 36.1119270324707, mae: 128.61907958984375, rmse: 6.005255699157715, psnr: 62.669097900390625
```

Let's take a look at what it created.  Notice it named the experiment `Debug` (that's the
default, but one can certainly specify that on the command line above).

You'll now have a `work_dirs/Debug/saved/` dir that looks something like this:

```
ls work_dirs/Debug/saved
inputs      metrics.npy preds       trues
```

# CUDA
Charliecloud supports running nvidia CUDA code with some effort.
First we must extract our container image.
Then we add CUDA secret sauce with `ch-fromhost --nvidia`
Finally we run on the modified image.

```
$ ch-convert simvp $HOME/simvp-cuda
discarding xattrs...
input:   ch-image  simvp
output:  dir       /home/rgoff/simvp-cuda
exporting ...
done
$ ch-fromhost --nvidia $HOME/simvp-cuda
(from /etc/ld.so.conf.d/988_cuda-12.conf:1 and /etc/ld.so.conf.d/000_cuda.conf:1)
(from /etc/ld.so.conf.d/cuda.conf:1 and /etc/ld.so.conf.d/000_cuda.conf:1)
(from /etc/ld.so.conf.d/gds-12-6.conf:1 and /etc/ld.so.conf.d/000_cuda.conf:1)
(from /etc/ld.so.conf.d/x86_64-linux-gnu.conf:4 and /etc/ld.so.conf.d/x86_64-linux-gnu.conf:3)
(from <builtin>:0 and /etc/ld.so.conf.d/x86_64-linux-gnu.conf:3)
(from <builtin>:0 and /etc/ld.so.conf.d/x86_64-linux-gnu.conf:3)
(from <builtin>:0 and <builtin>:0)
error: not a directory: /home/rgoff/simvp-cuda/usr/local/cuda/compat/lib
# I am ignoring the error it didn't seem to hurt me at execution

$ ch-run -w -b $HOME/data_boiling:/mnt/0 $HOME/simvp-cuda -- python /simvp/train_simvp_standalone.py --datafile_in /mnt/0/$MY_LOADER --pre_seq_length 10 --aft_seq_length 10 --device cuda --config_file /simvp-main/configs/SimVP_super_simple.py
```

Running from a flat directory is not a charliecloud best practice, this is done to simplify the example. You should take that directory and turn it back into a squash filesystem.

# Visualization
You can follow the original readme for visualiztion on the host