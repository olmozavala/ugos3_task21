# Satellite Data Assimilation to forecast SSH

## TODO
- [x] Only 1 channel output, it is usedc to compute the gradient and both are used in the loss function in `gradient` mode
- [x] Remove the gradient calculation from the data loader, it is computed as the model trains
- [ ] Make the noise added to the ssh proportional to the magnitude of the ssh
- [ ] Update the config file to add a `loss_mode` parameter to choose between `rmse` and `rmse_gradient`
- [ ] Update the README.md to reflect the new changes

This model is a UNet with upsampling layers.

## Training modes
The user must select a set number of previous days to use as input data (default is 7). The Gulf mask is always the last channel.
- `regular`: Input is (previous_days * 4 + gulf_mask, height, width) channels. Output is (1, height, width) channels. Output is the ssh (1, height, width) for the next day. The loss function is the mse between the predicted ssh and the target ssh only.
- `extended`: Input is (previous_days * 4 + the ssh of the 2 previous days + the gulf mask, height, width) channels. Output is (1, height, width) channels. Output is the ssh (1, height, width) for the next day. The loss function is the mse between the predicted ssh and the target ssh only.
- `gradient`: Input is (previous_days * 4 + the ssh of the 2 previous days + the gulf mask, height, width) channels. Output is (1, height, width) channels. Output is the predicted SSH (1, height, width) for the next day. The loss function incorporates the mse of both the predicted SSH and the gradient of the predicted SSH.
    $$ L = MSE(predicted_{ssh}, target_{ssh}) + MSE(grad(predicted_{ssh}), grad(target_{ssh})) $$



