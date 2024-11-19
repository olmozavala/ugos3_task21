# Satellite Data Assimilation to forecast SSH

## TODO
- [x] Only 1 channel output, it is usedc to compute the gradient and both are used in the loss function in `gradient` mode
- [x] Remove the gradient calculation from the data loader, it is computed as the model trains
- [ ] Make the noise added to the ssh proportional to the magnitude of the ssh
- [ ] Update the config file to add a `loss_mode` parameter to choose between `rmse` and `rmse_gradient`
- [x] Work on the implementation of an autoregressive model using the current model
    - [x] Create a new dataloder subclass that enables to pass the previous predictions as input and future predictions as target
    - [x] Update the model architecture to enable the input of previous predictions
    - **Moved to a separate folder** check `autoreg` folder
- [x] Update the README.md to reflect the new changes

This model is a UNet with upsampling layers.

## Training modes
The user must select a set number of previous days to use as input data (default is 7). The Gulf mask is always the last channel.
- `regular`: Input is (previous_days * 4 + gulf_mask, height, width) channels. Output is (1, height, width) channels. Output is the ssh (1, height, width) for the next day. The loss function is the mse between the predicted ssh and the target ssh only.
- `extended`: Input is (previous_days * 4 + the ssh of the 2 previous days + the gulf mask, height, width) channels. Output is (1, height, width) channels. Output is the ssh (1, height, width) for the next day. The loss function is the mse between the predicted ssh and the target ssh only.
- `gradient`: Input is (previous_days * 4 + the ssh of the 2 previous days + the gulf mask, height, width) channels. Output is (1, height, width) channels. Output is the predicted SSH (1, height, width) for the next day. The loss function incorporates the mse of both the predicted SSH and the gradient of the predicted SSH.
    $$ L = MSE(predicted_{ssh}, target_{ssh}) + MSE(grad(predicted_{ssh}), grad(target_{ssh})) $$

## Autoregressive model in construction
The new data loader class is `data_autoregressive.py`. It is used to train an autoregressive model that uses the previous predictions as input and the future predictions as target. It has a new flag `horizon_days` that specifies the number of days ahead we want to predict.

Its getting harder to make partial changes from the current model. I will move to a separtate folder to work in the autoregressive model. SInce It requieres a lot of changes and this will most likely break the current model, if they are not already broken. I will creating copies insde this new folrder an create a new `README.md` inside it to track the changes and the TODOs.