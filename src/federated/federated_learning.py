import sys
sys.path.append('../../src')

from collections import OrderedDict

from models.model import get_model


def get_federated_avg_model(remote_models, train_set_sizes):
    weights = OrderedDict()
    # compute the weighted average of the model weights/biases
    for i in range(len(remote_models)):
        # save the remote trained model
        local_model_updates = remote_models[i].state_dict()

        if len(weights) == 0:
            for layer in local_model_updates.keys():
                weights[layer] = local_model_updates[layer]*train_set_sizes[i]
        else:
            for layer in local_model_updates.keys():
                weights[layer] = local_model_updates[layer]*train_set_sizes[i]

    for layer in local_model_updates.keys():
        weights[layer] = local_model_updates[layer] / sum(train_set_sizes)

    # create an empty model and load the combined weights to it
    combined_model = get_model()
    combined_model.load_state_dict(weights)
    return combined_model
