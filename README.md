Pydrake RGBD Simulation
=================

Uses Chris Sweeney's Keras/CNN-based depth noise learner (see submodule) to produce RGBD simulations of scenes.

Significant reference to [this](https://github.com/keras-team/keras/tree/master/docker) Docker reference and [this](https://github.com/gizatt/drake_periscope_tutorial) Pydrake example. Those are better documented than this repo.

## Gist of how to run this:

### Using a hosted docker image
`make bash` should download the right image and launch a shell script. You can change `IMAGE_NAME` by editing the `Makefile`.

In your host environment, `pip install meshcat` and run `meshcat-server` in a separate terminal. This'll create a host-run visualizer. (I could adapt this to also run in the container, but it's hopefully as easy to have it run on the host...)

Then, `cd workspace` and `python rgbd_tabletop_sim`. Run with `--help` to see options.

### Rebuilding the Docker image
Copy an extracted Drake build (e.g. legitimate Drake binaries, or the `build/install` folder of a from-source Drake build) as a folder named `drake_install`, and then run `make build`.