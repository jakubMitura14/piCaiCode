from comet_ml import Optimizer

# We only need to specify the algorithm and hyperparameters to use:
config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {
        "x": {"type": "integer", "min": 1, "max": 5},
    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "loss",
        "objective": "minimize",
    },
}

# Next, create an optimizer, passing in the config:
# (You can leave out API_KEY if you already set it)
opt = Optimizer(config)