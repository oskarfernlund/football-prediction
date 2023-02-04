"""
Optimisation functions.
"""

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import tensorflow as tf

from gpflow.ci_utils import reduce_in_tests
from gpflow.models import SVGP

from src.base import RegressionData


# =============================================================================
#  GLOBAL VARIABLES
# =============================================================================

def optimise_svgp_with_adam(
    model: SVGP,
    data: RegressionData,
    lr: float = 0.001,
    batch_size: int = 100,
    iters: int = 1000
) -> list:
    """ Optimise a GPflow SVGP model using an ADAM optimiser.

    Args:
        model (SVGP) : Initialized GPflow SVGP model to optimise
        data (RegressionData) : Training data to optimise on
        lr (float) : Learning rate for the ADAM optimiser
        batch_size (int) : Batch size for training
        iters (int) : Number of training iterations

    Returns:
        elbos (list) : List of loss values at each training iteration
    """
    # Make sure the provided model is a GPflow SVGP model
    assert type(model) == SVGP

    # Create batched Tensorflow dataset 
    data = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(model.num_data)

    # Initialize ADAM optimiser
    elbos = []
    train_iter = iter(data.batch(batch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    # Define optimisation step
    @tf.function
    def optimisation_step():
        optimizer.minimize(training_loss, model.trainable_variables)

    # Run optimisation loop and log every 10th loss
    for step in range(reduce_in_tests(iters)):
        optimisation_step()
        elbo = -training_loss().numpy()
        elbos.append(elbo)
        
    return elbos
    